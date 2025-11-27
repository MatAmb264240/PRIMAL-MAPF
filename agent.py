import numpy as np
import tensorflow as tf

from buffer import ExperienceBuffer


class A3CAgent:
    def __init__(
        self,
        model,
        optimizer: tf.keras.optimizers.Optimizer,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        valid_coef: float = 0.1,
        blocking_coef: float = 0.5,     # ← DODANE!
    ):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma

        # współczynniki strat – PRIMAL-style
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.valid_coef = valid_coef
        self.blocking_coef = blocking_coef  # ← ZAPISANE W KLASIE!

        # bufor: trener podmienia go na merged_buffer
        self.buffer = ExperienceBuffer(gamma=gamma)

    # ------------------------------------------------------------
    #   ACTION SELECTION
    # ------------------------------------------------------------
    def choose_action(self, observation: dict, lstm_state=None):
        fov = observation["fov"]
        goal_vec = observation["goal_vec"]
        action_mask = observation.get("action_mask", None)

        fov_t = tf.expand_dims(tf.convert_to_tensor(fov, dtype=tf.float32), 0)
        goal_t = tf.expand_dims(tf.convert_to_tensor(goal_vec, dtype=tf.float32), 0)

        # LSTM stan
        if lstm_state is not None:
            hx, cx = lstm_state
            hx = tf.convert_to_tensor(hx, dtype=tf.float32)
            cx = tf.convert_to_tensor(cx, dtype=tf.float32)
            lstm_state = (
                hx if hx.shape[0] == 1 else hx[:1],
                cx if cx.shape[0] == 1 else cx[:1],
            )

        policy_logits, value, _, new_state = self.model(
            fov_t, goal_t, state=lstm_state, training=False
        )

        logits = policy_logits[0]
        value = float(value[0, 0].numpy())

        # maskowanie akcji
        if action_mask is not None:
            mask = tf.convert_to_tensor(action_mask, dtype=tf.float32)
            logits = logits + tf.math.log(mask + 1e-8)

        probs = tf.nn.softmax(logits).numpy()
        probs_sum = probs.sum()

        if probs_sum <= 0.0 or not np.isfinite(probs_sum):
            # fallback
            if action_mask is not None:
                valid = np.where(np.array(action_mask) > 0.0)[0]
                action = int(valid[0] if len(valid) == 0 else np.random.choice(valid))
            else:
                action = 0
        else:
            probs /= probs_sum
            action = int(np.random.choice(len(probs), p=probs))

        return action, value, new_state

    # ------------------------------------------------------------
    #   BACKPROP (IL + RL)
    # ------------------------------------------------------------
    def update_from_buffer(self, il_mode: bool = False):
        if len(self.buffer) == 0:
            return

        # wejścia
        fovs = np.stack([s["fov"] for s in self.buffer.states], axis=0)
        goals = np.stack([s["goal_vec"] for s in self.buffer.states], axis=0)
        action_masks = np.stack([s["action_mask"] for s in self.buffer.states], axis=0)

        # blocking label (PRIMAL-style)
        # blocking_targets = np.array(
        #     [s.get("blocking_label", 0.0) for s in self.buffer.states],
        #     dtype=np.float32,
        # ).reshape(-1, 1)

        actions = np.array(self.buffer.actions, dtype=np.int32)

        # returns i advantages
        returns, advantages = self.buffer.compute_returns_and_advantages()
        returns = returns.astype(np.float32)
        advantages = advantages.astype(np.float32)

        if not il_mode:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns_t = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages_t = tf.convert_to_tensor(advantages, dtype=tf.float32)

        with tf.GradientTape() as tape:
            (
                policy_logits,
                values_pred,
                # blocking_logits,
                valid_logits,
                _,
            ) = self.model(
                tf.convert_to_tensor(fovs, dtype=tf.float32),
                tf.convert_to_tensor(goals, dtype=tf.float32),
                state=None,
                training=True,
            )

            log_probs = tf.nn.log_softmax(policy_logits, axis=-1)
            probs = tf.nn.softmax(policy_logits, axis=-1)

            # log p(a|s)
            idx = tf.stack([tf.range(len(actions)), actions], axis=1)
            chosen_log_probs = tf.gather_nd(log_probs, idx)

            # Policy loss
            if il_mode:
                # BEHAVIOR CLONING (expert imitation)
                policy_loss = -tf.reduce_mean(chosen_log_probs)
            else:
                policy_loss = -tf.reduce_mean(chosen_log_probs * advantages_t)

            # Value loss
            value_loss = self.value_coef * tf.reduce_mean(
                tf.square(returns_t - tf.squeeze(values_pred, axis=-1))
            )

            # Entropia
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=-1))

            # Valid head
            valid_targets = tf.convert_to_tensor(action_masks, dtype=tf.float32)
            valid_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=valid_targets, logits=valid_logits
                )
            )

            # Blocking head
            # blocking_targets_t = tf.convert_to_tensor(blocking_targets, dtype=tf.float32)
            # blocking_loss = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(
            #         labels=blocking_targets_t, logits=blocking_logits
            #     )
            # )

            loss = (
                policy_loss
                + value_loss
                - self.entropy_coef * entropy
                + self.valid_coef * valid_loss
                # + self.blocking_coef * blocking_loss
            )

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, self.model.trainable_variables) if g is not None]
        self.optimizer.apply_gradients(grads_and_vars)

        self.buffer.clear()