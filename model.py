import tensorflow as tf
from tensorflow.keras import layers


class ACNet(tf.keras.Model):
    """
    Prosta sieć Actor-Critic inspirowana PRIMAL.
    """

    def __init__(self, num_actions: int = 5):
        super().__init__()
        self.num_actions = num_actions

        # CNN tower dla FOV
        self.conv1 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.conv2 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.conv3 = layers.Conv2D(128, 3, padding="same", activation="relu")
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2))

        self.conv4 = layers.Conv2D(128, 3, padding="same", activation="relu")
        self.conv5 = layers.Conv2D(256, 3, padding="same", activation="relu")
        self.conv6 = layers.Conv2D(256, 3, padding="same", activation="relu")
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2))

        self.flatten = layers.Flatten()
        self.fov_fc = layers.Dense(512, activation="relu")

        # wektor celu
        self.goal_fc = layers.Dense(32, activation="relu")

        # górka FC
        self.fc1 = layers.Dense(512, activation="relu")
        self.fc2 = layers.Dense(512, activation="relu")

        # ścieżka rezydualna
        self.res_proj = layers.Dense(512)

        # LSTM
        self.lstm = layers.LSTMCell(512)

        # wyjścia
        self.policy = layers.Dense(num_actions)
        self.value = layers.Dense(1)
        # self.blocking = layers.Dense(1)
        self.valid = layers.Dense(num_actions)

    def call(self, fov, goal_vec, state=None, training=False):
        x = tf.cast(fov, tf.float32)
        g = tf.cast(goal_vec, tf.float32)

        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fov_fc(x)  # [B, 512]

        # goal embedding
        g = self.goal_fc(g)  # [B, 32]

        # concat
        concat = tf.concat([x, g], axis=-1)  # [B, 544]

        # FC tower
        y = self.fc1(concat)
        y = self.fc2(y)

        # residual
        res = self.res_proj(concat)
        lstm_input = y + res  # [B, 512]

        # LSTM state
        if state is None:
            batch_size = tf.shape(lstm_input)[0]
            hx = tf.zeros([batch_size, 512], dtype=tf.float32)
            cx = tf.zeros([batch_size, 512], dtype=tf.float32)
        else:
            hx, cx = state

        # LSTMCell zwraca (output, [new_h, new_c])
        output, [hx, cx] = self.lstm(lstm_input, states=[hx, cx], training=training)

        policy_logits = self.policy(output)
        value = self.value(output)
        # blocking_logits = self.blocking(output)
        valid_logits = self.valid(output)

        return policy_logits, value, valid_logits, (hx, cx)

    def get_value(self, fov, goal_vec):
        fov_t = tf.convert_to_tensor(fov, dtype=tf.float32)
        goal_t = tf.convert_to_tensor(goal_vec, dtype=tf.float32)

        if len(fov_t.shape) == 3:
            fov_t = tf.expand_dims(fov_t, 0)
        if len(goal_t.shape) == 1:
            goal_t = tf.expand_dims(goal_t, 0)

        _, value, _, _ = self.call(fov_t, goal_t, state=None, training=False)
        return tf.squeeze(value, axis=-1).numpy()