import tensorflow as tf
from tensorflow.keras import layers

class ACNet(tf.keras.Model):
    def __init__(self, num_actions=5):
        super().__init__()

        # --- CNN tower ---
        self.conv1 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.conv2 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.conv3 = layers.Conv2D(128, 3, padding="same", activation="relu")
        self.pool1 = layers.MaxPool2D(pool_size=(2,2))  # 10x10 → 5x5

        self.conv4 = layers.Conv2D(128, 3, padding="same", activation="relu")
        self.conv5 = layers.Conv2D(256, 3, padding="same", activation="relu")
        self.conv6 = layers.Conv2D(256, 3, padding="same", activation="relu")
        self.pool2 = layers.MaxPool2D(pool_size=(2,2))  # 5x5 → 2x2

        self.flatten = layers.Flatten()
        self.fov_fc = layers.Dense(512, activation="relu")

        # --- Goal vector ---
        self.goal_fc = layers.Dense(32, activation="relu")

        # --- Fully connected tower ---
        self.fc1 = layers.Dense(512, activation="relu")
        self.fc2 = layers.Dense(512, activation="relu")

        # --- Residual projection ---
        self.res_proj = layers.Dense(512)

        # --- LSTM ---
        self.lstm = layers.LSTMCell(512)

        # --- Output heads ---

        self.policy = layers.Dense(num_actions)      # logits - prawdopodobienstwo wyboru jakiejś akcji
        self.value = layers.Dense(1)                 # V(s) - wartosc stanu ( czyli dla tego stanu będzie taka i taka nagroda i karea i potem to jest używane do korygowania tej sieci w A2C/A3C)
        self.blocking = layers.Dense(1)              # logits (sigmoid w lossie) - ocenia czy blokuje innych agentów
        self.valid = layers.Dense(num_actions)       # logits - które akcje są legalne w danym stanie

    @tf.function
    def call(self, fov, goal_vec, state=None, training=False):
        """
        fov: [B,10,10,4]
        goal_vec: [B,3]
        state: (hx, cx) or None
        """
        x = tf.cast(fov, tf.float32)

        # --- CNN tower ---
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fov_fc(x)  # [B,512]

        # --- Goal embedding ---
        g = self.goal_fc(goal_vec)  # [B,32]

        # --- Concat CNN + goal ---
        concat = tf.concat([x, g], axis=-1)  # [B,544]

        # --- FC tower ---
        y = self.fc1(concat)
        y = self.fc2(y)

        # --- Residual path ---
        res = self.res_proj(concat)

        lstm_input = y + res  # [B,512]

        # --- LSTM state init ---
        if state is None:
            batch = tf.shape(lstm_input)[0]
            hx = tf.zeros([batch, 512], dtype=tf.float32)
            cx = tf.zeros([batch, 512], dtype=tf.float32)
        else:
            hx, cx = state

        # --- LSTM forward ---
        (hx, cx) = self.lstm(lstm_input, states=[hx, cx], training=training)

        # --- Output heads ---
        policy_logits = self.policy(hx)
        value = self.value(hx)
        blocking_logits = self.blocking(hx)
        valid_logits = self.valid(hx)

        return policy_logits, value, blocking_logits, valid_logits, (hx, cx)
