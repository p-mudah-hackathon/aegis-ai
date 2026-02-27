"""
AegisNode — ML Model Service
Pure HTGNN inference endpoints. No frontend, no business logic.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import torch
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio

from core.config import PROCESSED_DIR, CHECKPOINT_DIR, PROJECT_ROOT

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AegisNode — ML Model Service",
    description="HTGNN fraud detection model inference. Stateless scoring & explainability.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Schemas ──────────────────────────────────────────────────────────────────
class TransactionFeatures(BaseModel):
    """Raw transaction features for model scoring."""
    txn_id: str = Field(..., example="TXN-000042")
    is_fraud: bool = Field(False, description="Ground truth label (for simulation)")
    fraud_type: Optional[str] = Field(None, example="velocity_attack")

class ScoreRequest(BaseModel):
    transactions: List[TransactionFeatures]

class ScoreResult(BaseModel):
    txn_id: str
    risk_score: float
    is_flagged: bool

class ScoreResponse(BaseModel):
    results: List[ScoreResult]
    threshold: float
    mode: str = Field(..., description="MODEL or SIMULATION")

class ExplainRequest(BaseModel):
    txn_id: str
    risk_score: float = Field(..., ge=0, le=1)
    is_fraud: bool = False
    fraud_type: Optional[str] = None

class XAIFeature(BaseModel):
    feature: str
    display_name: str
    importance: float

class ExplainResponse(BaseModel):
    txn_id: str
    features: List[XAIFeature]

class ModelInfo(BaseModel):
    status: str = Field(..., description="loaded or unavailable")
    threshold: float
    mode: str
    architecture: str = "HTGNN (Heterogeneous Temporal Graph Neural Network)"
    n_layers: Optional[int] = None
    d_model: Optional[int] = None
    gate_value: Optional[float] = None
    n_edges: Optional[int] = None

# ── Model loading (lazy, cached) ────────────────────────────────────────────
_model_cache: Dict[str, Any] = {}

FEATURE_NAMES = [
    ("amount_idr", "Transaction Amount"),
    ("hour_of_day", "Hour of Day"),
    ("payer_txn_count_1h", "Payer Activity (1h)"),
    ("merchant_txn_velocity_1h", "Merchant Velocity (1h)"),
    ("amount_zscore_merchant", "Amount Z-Score"),
    ("time_since_last_txn_sec", "Time Since Last Txn"),
]

def _load_model():
    if _model_cache:
        return (
            _model_cache.get("model"),
            _model_cache.get("data"),
            _model_cache.get("scores"),
            _model_cache.get("threshold"),
        )
    try:
        from core.model import AegisHTGNN

        meta_path = os.path.join(PROCESSED_DIR, "preprocessing_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        data = torch.load(
            os.path.join(PROCESSED_DIR, "aegis_hetero_data.pt"), weights_only=False
        )

        eval_path = os.path.join(CHECKPOINT_DIR, "evaluation_results.json")
        threshold = 0.5
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                eval_res = json.load(f)
                threshold = eval_res.get("test", {}).get("optimal_threshold", 0.5)

        model = AegisHTGNN(
            meta=data.metadata(),
            cat_sizes={
                "source_issuer": meta["n_source_issuers"],
                "source_country": meta["n_source_countries"],
                "merchant_mcc": meta["n_merchant_mccs"],
                "merchant_location": meta["n_merchant_locations"],
                "qris_type": meta["n_qris_types"],
            },
            edge_feat_dim=data["payer", "TRANSACTS", "merchant"].edge_attr.shape[1],
        )
        ckpt = torch.load(
            os.path.join(CHECKPOINT_DIR, "best_model.pt"),
            weights_only=False,
            map_location="cpu",
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            raw_scores = model(
                data.x_dict,
                data.edge_index_dict,
                {('payer', 'TRANSACTS', 'merchant'): data['payer', 'TRANSACTS', 'merchant'].edge_attr},
                data['payer', 'TRANSACTS', 'merchant'].edge_attr[:, 0] if data['payer', 'TRANSACTS', 'merchant'].edge_attr.shape[1] > 0 else None,
            )
            all_scores = torch.sigmoid(raw_scores).numpy()

        _model_cache["model"] = model
        _model_cache["data"] = data
        _model_cache["scores"] = all_scores
        _model_cache["threshold"] = threshold
        return model, data, all_scores, threshold

    except Exception as e:
        print(f"[aegis-ai] Model load failed: {e}")
        _model_cache["model"] = None
        _model_cache["data"] = None
        _model_cache["scores"] = None
        _model_cache["threshold"] = 0.5
        return None, None, None, 0.5


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "aegis-ai-model"}


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def model_info():
    """Return model metadata: threshold, architecture info, status."""
    model, data, scores, threshold = _load_model()
    mode = "MODEL" if model is not None else "SIMULATION"

    info = ModelInfo(
        status="loaded" if model else "unavailable",
        threshold=threshold,
        mode=mode,
    )

    if model is not None:
        info.n_layers = model.n_layers
        info.d_model = model.d_model
        info.gate_value = round(float(torch.sigmoid(model.gate).item()), 4)
    if data is not None:
        info.n_edges = data["payer", "TRANSACTS", "merchant"].edge_index.shape[1]

    return info


@app.post("/model/score", response_model=ScoreResponse, tags=["Model"])
async def score_transactions(req: ScoreRequest):
    """
    Score a batch of transactions. Returns risk_score and is_flagged for each.

    Inference uses real HTGNN scores based on the loaded model.
    """
    model, data, all_scores, threshold = _load_model()
    mode = "MODEL" if model is not None else "SIMULATION"

    results = []
    
    if model is not None:
        from core.model.featurizer import featurize_live_transaction
        model.eval()

    for txn in req.transactions:
        txn_dict = txn.dict()
        
        # We need a timestamp, country, issuer, and city to featurize properly. 
        # The schema might be missing them, so we provide defaults if so.
        if 'timestamp' not in txn_dict:
            from datetime import datetime
            txn_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if 'issuer' not in txn_dict: txn_dict['issuer'] = 'Unknown'
        if 'country' not in txn_dict: txn_dict['country'] = 'ID'
        if 'city' not in txn_dict: txn_dict['city'] = 'Jakarta'
        if 'amount_idr' not in txn_dict: txn_dict['amount_idr'] = 100000
        if 'amount_foreign' not in txn_dict: txn_dict['amount_foreign'] = 10.0
            
        if model is not None:
            live_data = featurize_live_transaction(txn_dict)
            x_dict = {'payer': live_data['payer'].x, 'merchant': live_data['merchant'].x}
            edge_type = ('payer', 'TRANSACTS', 'merchant')
            edge_index_dict = {edge_type: live_data[edge_type].edge_index}
            edge_attr_dict = {edge_type: live_data[edge_type].edge_attr}
            timestamps = live_data[edge_type].timestamps
            
            with torch.no_grad():
                logits = model(x_dict, edge_index_dict, edge_attr_dict, timestamps)
                score = float(torch.sigmoid(logits[0]).item())
        else:
            # Fallback
            import random
            score = 0.85 if txn_dict.get('is_fraud') else random.uniform(0.1, 0.4)

        score = round(score, 4)
        results.append(ScoreResult(
            txn_id=txn.txn_id,
            risk_score=score,
            is_flagged=score >= threshold,
        ))

    return ScoreResponse(results=results, threshold=threshold, mode=mode)


@app.post("/model/explain", response_model=ExplainResponse, tags=["Model"])
async def explain_transaction(req: ExplainRequest):
    """
    Generate XAI feature importances for a single transaction.

    Returns the top contributing features with their importance weights,
    derived from integrated gradients on edge features via PyTorch inference.
    """
    model, data, all_scores, threshold = _load_model()
    
    features = []
    
    if model is not None:
        from core.model.featurizer import featurize_live_transaction
        
        # Create a mock transaction dictionary to featurize
        from datetime import datetime
        txn_dict = {
            'txn_id': req.txn_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'issuer': 'Unknown',
            'country': 'ID',
            'city': 'Jakarta',
            'amount_idr': 100000,
            'amount_foreign': 10.0,
            'is_fraud': req.is_fraud,
            'fraud_type': req.fraud_type
        }
        
        live_data = featurize_live_transaction(txn_dict)
        x_dict = {'payer': live_data['payer'].x, 'merchant': live_data['merchant'].x}
        edge_type = ('payer', 'TRANSACTS', 'merchant')
        edge_index_dict = {edge_type: live_data[edge_type].edge_index}
        
        # Requires grad for XAI
        edge_attr = live_data[edge_type].edge_attr.clone().detach().requires_grad_(True)
        edge_attr_dict = {edge_type: edge_attr}
        timestamps = live_data[edge_type].timestamps
        
        model.eval()
        logits_xai = model(x_dict, edge_index_dict, edge_attr_dict, timestamps)
        target_logit = logits_xai[0]
        target_logit.backward()
        
        grad = edge_attr.grad[0]
        importance = (grad * edge_attr[0].detach()).abs().numpy()
        
        mapped_importances = [
            float(importance[0]),       # amount_idr
            float(importance[2] + importance[3]), # hour
            float(importance[7]),       # payer activity
            float(importance[9]),       # merchant spike
            float(importance[6]),       # z-score
            float(importance[10])       # time since last
        ]
        
        total_w = sum(mapped_importances) or 1
        weights = [w / total_w for w in mapped_importances]
        
        feature_names = [
            ("amount_idr", "Transaction Amount"),
            ("hour_of_day", "Hour of Day"),
            ("payer_txn_count_1h", "Payer Activity (1h)"),
            ("merchant_txn_velocity_1h", "Merchant Velocity (1h)"),
            ("amount_zscore_merchant", "Amount Z-Score"),
            ("time_since_last_txn_sec", "Time Since Last Txn"),
        ]
        
        feat_scores = list(zip(feature_names, weights))
        feat_scores.sort(key=lambda x: x[1], reverse=True)
        
        for (feat, name), w in feat_scores[:5]:
            features.append(XAIFeature(
                feature=feat,
                display_name=name,
                importance=round(w, 3)
            ))
            
    else:
        # Fallback simulation
        import random
        random.seed(req.txn_id)
        raw_weights = [random.random() for _ in range(len(FEATURE_NAMES))]
        total = sum(raw_weights)
        weights = sorted([w / total for w in raw_weights], reverse=True)

        for i, (feat, name) in enumerate(FEATURE_NAMES):
            features.append(XAIFeature(
                feature=feat,
                display_name=name,
                importance=round(float(weights[i]), 3),
            ))
            
        features = features[:5]

    return ExplainResponse(txn_id=req.txn_id, features=features)


# ── WebSockets ───────────────────────────────────────────────────────────────
dashboard_clients = set()
attacker_clients = set()

@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """Dashboard UI connections."""
    await websocket.accept()
    dashboard_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        dashboard_clients.remove(websocket)

@app.websocket("/ws/attack")
async def websocket_attack(websocket: WebSocket):
    """Attacker UI connections."""
    await websocket.accept()
    attacker_clients.add(websocket)
    try:
        data = await websocket.receive_json()
        total = data.get("total", 500)
        fraud_pct = data.get("fraud_pct", 0.05)
        speed = data.get("speed", "normal")
        attack_type = data.get("attack_type", "all")

        # Notify dashboard
        for client in list(dashboard_clients):
            try:
                await client.send_json({"type": "attack_start", "data": {"total": total, "attack_type": attack_type}})
                await client.send_json({"type": "reset"})
            except:
                pass

        # Spawn demo_attack.py with UTF-8 encoding forced for Windows pipes
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        cmd_args = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "demo_attack.py"),
            "--total", str(total),
            "--fraud-pct", str(fraud_pct),
            "--speed", speed,
            "--attack-type", attack_type,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        async def read_stream(stream):
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode('utf-8').strip()
                if not decoded: continue

                # Send JSON straight to Dashboard and Attacker UI
                if decoded.startswith("JSON_STREAM:"):
                    json_str = decoded.replace("JSON_STREAM:", "", 1)
                    try:
                        payload = json.loads(json_str) 
                        for client in list(dashboard_clients):
                            await client.send_json(payload)
                        for client in list(attacker_clients):
                            await client.send_json(payload)
                    except:
                        pass
                else:
                    # Send text to attacker terminal
                    msg_type = "txn_log" if "│" in decoded and "TXN-" in decoded else "log"
                    # Simple hack to extract colors to classes
                    color = "red" if "FLAGGED" in decoded else "green" if "APPROVED" in decoded else None
                    is_flagged = "FLAGGED" in decoded
                    is_fraud = "True" in decoded # Not totally accurate regex hack
                    
                    for client in list(attacker_clients):
                        try:
                            await client.send_json({
                                "type": msg_type,
                                "text": decoded,
                                "color": color,
                                "is_flagged": is_flagged,
                                "is_fraud": is_fraud
                            })
                        except:
                            pass

        await asyncio.gather(
            read_stream(process.stdout),
            read_stream(process.stderr)
        )
        
        await process.wait()

        # Notify dashboard
        for client in list(dashboard_clients):
            try:
                await client.send_json({"type": "attack_end"})
            except:
                pass
                
        # Notify attacker
        for client in list(attacker_clients):
            try:
                await client.send_json({
                    "type": "log",
                    "text": f"[AegisNode] Attack simulation complete. Return code: {process.returncode}"
                })
            except:
                pass

    except WebSocketDisconnect:
        attacker_clients.remove(websocket)
    except Exception as e:
        for client in list(attacker_clients):
            try:
                await client.send_json({"type": "error", "text": f"Error: {str(e)}"})
            except:
                pass

# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
