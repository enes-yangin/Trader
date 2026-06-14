import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel
from utils.config import LSTM_PARAMS, SEQ_LEN


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def to_loader(X, y, batch_size, shuffle=True):
    tx = torch.FloatTensor(X).to(DEV)
    ty = torch.FloatTensor(y).to(DEV)
    ds = TensorDataset(tx, ty)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class LSTMNet(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, drop):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=drop if n_layers > 1 else 0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hid_dim // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class LSTMModel(BaseModel):
    def __init__(self, seq_len=SEQ_LEN, **kw):
        super().__init__(name="LSTM")
        self.p = {**LSTM_PARAMS, **kw}
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        self.net = None
        self.history = {"train_loss": [], "val_loss": []}

    def _build_net(self, in_dim):
        self.net = LSTMNet(
            in_dim=in_dim,
            hid_dim=self.p["hidden_dim"],
            n_layers=self.p["num_layers"],
            drop=self.p["dropout"],
        ).to(DEV)

    def _prep(self, X, y=None, fit_scaler=False):
        if fit_scaler:
            Xs = self.scaler.fit_transform(X)
        else:
            Xs = self.scaler.transform(X)
        if y is not None:
            return make_sequences(Xs, y, self.seq_len)
        return self._prep_seq(Xs)

    def _prep_seq(self, Xs):
        Xs_seq = []
        for i in range(len(Xs) - self.seq_len):
            Xs_seq.append(Xs[i:i + self.seq_len])
        return np.array(Xs_seq)

    def train(self, X_tr, y_tr, X_val=None, y_val=None):
        Xsq, ysq = self._prep(X_tr, y_tr, fit_scaler=True)
        self._build_net(Xsq.shape[2])

        bs = self.p["batch_size"]
        tr_loader = to_loader(Xsq, ysq, bs, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            Xv, yv = self._prep(X_val, y_val)
            val_loader = to_loader(Xv, yv, bs, shuffle=False)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.p["lr"])
        loss_fn = nn.MSELoss()
        self.history = {"train_loss": [], "val_loss": []}

        for ep in range(self.p["epochs"]):
            self.net.train()
            ep_loss = 0
            n = 0
            for xb, yb in tr_loader:
                opt.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item() * len(xb)
                n += len(xb)
            self.history["train_loss"].append(ep_loss / n)

            if val_loader:
                vl = self._val_loss(val_loader, loss_fn)
                self.history["val_loss"].append(vl)

        self.trained = True
        out = {"train": self._eval_from_raw(X_tr, y_tr)}
        if X_val is not None and y_val is not None:
            out["val"] = self._eval_from_raw(X_val, y_val)
        out["history"] = self.history
        return out

    def _val_loss(self, loader, loss_fn):
        self.net.eval()
        total, n = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                pred = self.net(xb)
                total += loss_fn(pred, yb).item() * len(xb)
                n += len(xb)
        return total / n

    def predict(self, X):
        self.net.eval()
        Xs = self._prep(X)
        tx = torch.FloatTensor(Xs).to(DEV)
        with torch.no_grad():
            preds = self.net(tx).cpu().numpy()
        return preds.flatten()

    def predict_last(self, X):
        self.net.eval()
        Xs = self.scaler.transform(X)
        if len(Xs) < self.seq_len:
            return 0.0
        seq = Xs[-self.seq_len:]
        tx = torch.FloatTensor(seq).unsqueeze(0).to(DEV)
        with torch.no_grad():
            p = self.net(tx).cpu().item()
        return p

    def _eval_from_raw(self, X, y):
        Xsq, ysq = self._prep(X, y)
        tx = torch.FloatTensor(Xsq).to(DEV)
        self.net.eval()
        with torch.no_grad():
            preds = self.net(tx).cpu().numpy().flatten()
        mse = float(np.mean((ysq - preds) ** 2))
        mae = float(np.mean(np.abs(ysq - preds)))
        ss_res = np.sum((ysq - preds) ** 2)
        ss_tot = np.sum((ysq - np.mean(ysq)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": mae, "r2": r2}
