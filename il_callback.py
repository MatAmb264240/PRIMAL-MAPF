import torch
from stable_baselines3.common.callbacks import BaseCallback

# RNNStates w sb3_contrib bywa w różnych miejscach zależnie od wersji
try:
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
except Exception:
    try:
        from sb3_contrib.common.recurrent.policies import RNNStates  # rzadki fallback
    except Exception:
        RNNStates = None


class OnlineILCallback(BaseCallback):
    def __init__(
        self,
        expert_loader_fn,
        il_coef=0.05,
        every_n_steps=50_000,
        verbose=0,
    ):
        super().__init__(verbose)
        self.expert_loader_fn = expert_loader_fn
        self.il_coef = il_coef
        self.every_n_steps = every_n_steps

    @staticmethod
    def _get_batch_size_and_device(obs):
        # obs może być tensorem albo dict tensora
        if isinstance(obs, dict):
            first = next(iter(obs.values()))
            return first.shape[0], first.device
        return obs.shape[0], obs.device

    @staticmethod
    def _zero_lstm_state(lstm_module, batch_size, device):
        """
        Zwraca (h, c) o kształcie (num_layers, batch, hidden_size)
        """
        num_layers = lstm_module.num_layers
        hidden_size = lstm_module.hidden_size
        h = torch.zeros((num_layers, batch_size, hidden_size), device=device)
        c = torch.zeros((num_layers, batch_size, hidden_size), device=device)
        return (h, c)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every_n_steps != 0:
            return True

        obs, expert_actions = self.expert_loader_fn()

        policy = self.model.policy
        optimizer = policy.optimizer

        batch_size, device = self._get_batch_size_and_device(obs)

        # WAŻNE: sb3_contrib w Twojej wersji robi (1.0 - episode_start), więc musi być float
        episode_starts = torch.ones((batch_size,), dtype=torch.float32, device=device)

        # === MUSI istnieć, bo w Twoim stacktrace jest self.lstm_actor ===
        if not hasattr(policy, "lstm_actor") or not hasattr(policy, "lstm_critic"):
            raise AttributeError(
                "Twoja policy nie ma lstm_actor/lstm_critic. "
                "Wypisz dir(model.policy) i sprawdź jak nazywają się moduły LSTM."
            )

        # Zbuduj zero-states dla actor i critic
        pi_state = self._zero_lstm_state(policy.lstm_actor, batch_size, device)
        vf_state = self._zero_lstm_state(policy.lstm_critic, batch_size, device)

        # Opakowanie w RNNStates jeśli dostępne
        if RNNStates is not None:
            lstm_states = RNNStates(pi=pi_state, vf=vf_state)
        else:
            # fallback: czasem przechodzi jako tuple (pi, vf)
            lstm_states = (pi_state, vf_state)

        # ==== KLUCZ: cudnn RNN backward wymaga train() ====
        policy.train()

        optimizer.zero_grad(set_to_none=True)

        # evaluate_actions to oficjalna ścieżka SB3 (log_prob policzony jak w PPO)
        _, log_prob, _ = policy.evaluate_actions(
            obs,
            expert_actions,
            lstm_states=lstm_states,
            episode_starts=episode_starts,
        )

        il_loss = -log_prob.mean()
        loss = self.il_coef * il_loss

        loss.backward()
        optimizer.step()

        # wracamy do eval, bo PPO zbiera rollouty w eval
        policy.eval()

        if self.verbose:
            print(f"[IL] step={self.num_timesteps}, loss={il_loss.item():.4f}")

        return True