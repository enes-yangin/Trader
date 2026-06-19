from typing import Optional, Tuple, Dict, List, Any, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel
from utils.config import MODEL
from utils.types import TrainResult, MetricsDict


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = MODEL.seq_len) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    tx = torch.FloatTensor(X).to(DEV)
    ty = torch.FloatTensor(y).to(DEV)
    ds = TensorDataset(tx, ty)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class LSTMNet(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, n_layers: int, drop: float):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class LSTMModel(BaseModel):
    def __init__(self, seq_len: int = MODEL.seq_len, **kw: Any):
        super().__init__(name="LSTM")
        self.p = {**MODEL.lstm_params, **kw}
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        self.net: Optional[LSTMNet] = None
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def _build_net(self, in_dim: int) -> None:
        self.net = LSTMNet(
            in_dim=in_dim,
            hid_dim=self.p["hidden_dim"],
            n_layers=self.p["num_layers"],
            drop=self.p["dropout"],
        ).to(DEV)

    def _prep(self, X: np.ndarray, y: Optional[np.ndarray] = None,
              fit_scaler: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if fit_scaler:
            Xs = self.scaler.fit_transform(X)
        else:
            Xs = self.scaler.transform(X)
        if y is not None:
            return make_sequences(Xs, y, self.seq_len)
        return self._prep_seq(Xs)

    def _prep_seq(self, Xs: np.ndarray) -> np.ndarray:
        Xs_seq = []
        for i in range(len(Xs) - self.seq_len):
            Xs_seq.append(Xs[i:i + self.seq_len])
        return np.array(Xs_seq)

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> TrainResult:
        Xsq, ysq = self._prep(X_tr, y_tr, fit_scaler=True)
        self._build_net(Xsq.shape[2])
        self.n_features_ = Xsq.shape[2]
        assert self.net is not None
        net = self.net

        bs = self.p["batch_size"]
        tr_loader = to_loader(Xsq, ysq, bs, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            Xv, yv = self._prep(X_val, y_val)
            val_loader = to_loader(Xv, yv, bs, shuffle=False)

        opt = torch.optim.Adam(net.parameters(), lr=self.p["lr"],
                                weight_decay=self.p.get("weight_decay", 0.0))
        loss_fn = nn.MSELoss()
        self.history = {"train_loss": [], "val_loss": []}

        patience = self.p.get("patience", 0)
        best_val = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_epoch = 0
        no_improve = 0

        for ep in range(self.p["epochs"]):
            net.train()
            ep_loss = 0.0
            n = 0
            for xb, yb in tr_loader:
                opt.zero_grad()
                pred = net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item() * len(xb)
                n += len(xb)
            self.history["train_loss"].append(ep_loss / n)

            if val_loader:
                vl = self._val_loss(val_loader, loss_fn)
                self.history["val_loss"].append(vl)
                if vl < best_val - 1e-9:
                    best_val = vl
                    best_epoch = ep
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if patience > 0 and no_improve >= patience:
                        break

        if best_state is not None:
            net.load_state_dict(best_state)

        self.trained = True
        out: TrainResult = {"train": self._eval_from_raw(X_tr, y_tr)}
        if X_val is not None and y_val is not None:
            out["val"] = self._eval_from_raw(X_val, y_val)
            out["best_epoch"] = best_epoch
        out["history"] = self.history
        return out

    def _val_loss(self, loader: DataLoader, loss_fn: nn.Module) -> float:
        assert self.net is not None
        net = self.net
        net.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                pred = net(xb)
                total += loss_fn(pred, yb).item() * len(xb)
                n += len(xb)
        return total / n

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_ready(X)
        assert self.net is not None
        net = self.net
        net.eval()
        Xs = self._prep(X)
        if len(Xs) == 0:
            return np.array([], dtype=float)
        tx = torch.FloatTensor(Xs).to(DEV)
        with torch.no_grad():
            preds = net(tx).cpu().numpy()
        return preds.flatten()

    def predict_last(self, X: np.ndarray) -> float:
        self._check_ready(X)
        assert self.net is not None
        net = self.net
        net.eval()
        Xs = self.scaler.transform(X)
        if len(Xs) < self.seq_len:
            return 0.0
        seq = Xs[-self.seq_len:]
        tx = torch.FloatTensor(seq).unsqueeze(0).to(DEV)
        with torch.no_grad():
            p = net(tx).cpu().item()
        return p

    def _eval_from_raw(self, X: np.ndarray, y: np.ndarray) -> MetricsDict:
        Xsq, ysq = self._prep(X, y)
        if len(Xsq) == 0:
            return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 0.0}
        assert self.net is not None
        net = self.net
        tx = torch.FloatTensor(Xsq).to(DEV)
        net.eval()
        with torch.no_grad():
            preds = net(tx).cpu().numpy().flatten()
        mse = float(np.mean((ysq - preds) ** 2))
        mae = float(np.mean(np.abs(ysq - preds)))
        ss_res = np.sum((ysq - preds) ** 2)
        ss_tot = np.sum((ysq - np.mean(ysq)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": mae, "r2": r2}

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> MetricsDict:
        return self._eval_from_raw(X, y)
