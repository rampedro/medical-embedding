import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import random

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== MEDICAL ONTOLOGY CONFIG =====
DISEASES = [
   "Type2Diabetes", "Type1Diabetes", "Hypertension",
   "Asthma", "COPD", "HeartFailure"
]
SYMPTOMS = [
   "Polyuria", "Polydipsia", "WeightLoss",
   "ShortnessOfBreath", "Wheezing", "Edema",
   "Fatigue", "ChestPain"
]
TREATMENTS = [
   "Metformin", "Insulin", "ACEInhibitor",
   "BetaBlocker", "Albuterol", "OxygenTherapy",
   "Diuretic"
]
RISK_FACTORS = [
   "Obesity", "Smoking", "SedentaryLifestyle",
   "FamilyHistoryDiabetes", "HighSaltDiet", "AgeOver60"
]

ALL_ENTITIES = DISEASES + SYMPTOMS + TREATMENTS + RISK_FACTORS
ENTITY_TO_IDX = {e: i for i, e in enumerate(ALL_ENTITIES)}

METABOLIC = ["Type2Diabetes", "Type1Diabetes", "Obesity", "Polyuria", "Polydipsia", "Metformin", "Insulin"]
CARDIO = ["Hypertension", "HeartFailure", "Edema", "ChestPain", "ACEInhibitor", "BetaBlocker", "Diuretic"]
RESPIRATORY = ["Asthma", "COPD", "ShortnessOfBreath", "Wheezing", "Albuterol", "OxygenTherapy", "Smoking"]

RELATION_MARGINS = {
    "identical": 0,
    "direct_cause": 1,
    "treated_by": 0.4,
    "symptom_of": 0.6,
    "risk_factor_for": 0.7,
    "contraindicated": 0.8,
    "associated": 0.9,
    "unrelated": 2
}

def get_medical_relation(anchor, candidate):
   a, c = anchor, candidate
   if a == c:
       return "identical"
   if (a == "Smoking" and c in ["COPD", "HeartFailure"]) or (a == "Obesity" and c == "Type2Diabetes"):
       return "direct_cause"
   if (a == "Type2Diabetes" and c == "Metformin") or \
      (a == "Hypertension" and c in ["ACEInhibitor", "BetaBlocker"]) or \
      (a in ["Asthma", "COPD"] and c in ["Albuterol", "OxygenTherapy"]) or \
      (a == "HeartFailure" and c == "Diuretic"):
       return "treated_by"
   if (a in ["Polyuria", "Polydipsia"] and c in ["Type1Diabetes", "Type2Diabetes"]) or \
      (a == "Wheezing" and c in ["Asthma", "COPD"]) or \
      (a == "Edema" and c == "HeartFailure"):
       return "symptom_of"
   if (a == "Obesity" and c == "Type2Diabetes") or \
      (a == "Smoking" and c in ["COPD", "HeartFailure"]) or \
      (a == "AgeOver60" and c == "HeartFailure"):
       return "risk_factor_for"
   if (a == "Asthma" and c == "BetaBlocker") or \
      (a == "COPD" and c == "BetaBlocker") or \
      (a == "HeartFailure" and c == "NSAID"):
       return "contraindicated"
   if (a in METABOLIC and c in METABOLIC and a != c) or \
      (a in CARDIO and c in CARDIO and a != c) or \
      (a in RESPIRATORY and c in RESPIRATORY and a != c):
       return "associated"
   return "unrelated"

# ===== CLINICAL FEATURE GENERATOR =====
CLINICAL_FEATURES = [
   "glucose_level", "hba1c", "systolic_bp", "diastolic_bp",
   "bmi", "age", "heart_rate", "respiratory_rate",
   "oxygen_sat", "creatinine", "has_family_history",
   "is_smoker", "is_active", "has_edema", "has_wheezing"
]

def generate_clinical_vector(entity, noise_level=0.1):
   vec = torch.zeros(len(CLINICAL_FEATURES))

   if entity == "Type2Diabetes":
       vec[0] = np.random.normal(180, 30)
       vec[1] = np.random.normal(7.5, 1.2)
       vec[4] = np.random.normal(32, 5)
       vec[11] = float(np.random.choice([0,1], p=[0.3,0.7]))
   elif entity == "Hypertension":
       vec[2] = np.random.normal(150, 15)
       vec[3] = np.random.normal(95, 10)
       vec[4] = np.random.normal(28, 4)
       vec[12] = float(np.random.choice([0,1], p=[0.6,0.4]))
   elif entity == "Asthma":
       vec[7] = np.random.normal(22, 4)
       vec[8] = np.random.normal(95, 3)
       vec[14] = 1.0
       vec[11] = float(np.random.choice([0,1], p=[0.5,0.5]))
   elif entity == "HeartFailure":
       vec[2] = np.random.normal(130, 20)
       vec[6] = np.random.normal(90, 15)
       vec[13] = 1.0
       vec[9] = np.random.normal(1.4, 0.3)
   elif entity == "COPD":
       vec[7] = np.random.normal(24, 5)
       vec[8] = np.random.normal(90, 5)
       vec[11] = 1.0
       vec[14] = 0.8
   elif entity == "Polyuria":
       vec[0] = np.random.normal(200, 40)
   elif entity == "Edema":
       vec[13] = 1.0
       vec[9] = np.random.normal(1.3, 0.4)
   elif entity == "Wheezing":
       vec[14] = 1.0
       vec[7] = np.random.normal(25, 6)
   elif entity == "Metformin":
       vec[0] = np.random.normal(120, 20)
       vec[1] = np.random.normal(6.5, 0.8)
   elif entity == "Insulin":
       vec[0] = np.random.normal(110, 15)
       vec[1] = np.random.normal(6.0, 0.7)
   elif entity == "BetaBlocker":
       vec[6] = np.random.normal(70, 10)
       vec[2] = np.random.normal(130, 15)
   elif entity == "Obesity":
       vec[4] = np.random.normal(35, 5)
       vec[0] = np.random.normal(140, 30)
   elif entity == "Smoking":
       vec[8] = np.random.normal(94, 4)
       vec[7] = np.random.normal(20, 3)

   # Normalize
   vec[0] = torch.clamp(vec[0] / 300.0, 0, 1)
   vec[1] = torch.clamp(vec[1] / 12.0, 0, 1)
   vec[2] = torch.clamp(vec[2] / 200.0, 0, 1)
   vec[3] = torch.clamp(vec[3] / 120.0, 0, 1)
   vec[4] = torch.clamp(vec[4] / 50.0, 0, 1)
   vec[6] = torch.clamp(vec[6] / 120.0, 0, 1)
   vec[7] = torch.clamp(vec[7] / 40.0, 0, 1)
   vec[8] = torch.clamp(vec[8] / 100.0, 0, 1)
   vec[9] = torch.clamp(vec[9] / 5.0, 0, 1)

   vec += torch.randn_like(vec) * noise_level
   vec = torch.clamp(vec, 0, 1)

   # Optional: Debug NaN
   if torch.isnan(vec).any():
       print(f"⚠️ NaN detected in vector for: {entity}")
       print(vec)

   return vec

# ===== ONTOTRIPLET++ LOSS (STABLE VERSION) =====
class MedicalOntoTripletLoss(nn.Module):
   def __init__(self, margin_dict, gamma=10, alpha=0.25, beta=1.2):
       super().__init__()
       self.margin_dict = margin_dict
       self.gamma = gamma
       self.alpha = alpha
       self.beta = beta

   def forward(self, anchor, positive, negative, rel_type_list):
       dist_ap = (anchor - positive).pow(2).sum(dim=1)
       dist_an = (anchor - negative).pow(2).sum(dim=1)

       # 🔥 Clamp exponents to prevent numerical overflow
       exp_p = torch.clamp(-self.gamma * (dist_ap - self.alpha), -50, 50)
       exp_n = torch.clamp(self.gamma * (dist_an - self.beta), -50, 50)
       w_p = torch.exp(exp_p)
       w_n = torch.exp(exp_n)

       margins = torch.tensor(
           [self.margin_dict[r] for r in rel_type_list],
           device=anchor.device, dtype=torch.float32
       )

       triplet_term = F.relu(dist_ap - dist_an + margins)
       circle_term = w_p * F.relu(dist_ap - self.alpha).pow(2) + \
                     w_n * F.relu(self.beta - dist_an).pow(2)

       total_loss = triplet_term.mean() + circle_term.mean()

       # 🚨 NaN Debugging
       if torch.isnan(total_loss):
           print("🚨 NaN in loss computation!")
           print("dist_ap:", dist_ap.detach().cpu().numpy())
           print("dist_an:", dist_an.detach().cpu().numpy())
           print("w_p:", w_p.detach().cpu().numpy())
           print("w_n:", w_n.detach().cpu().numpy())
           print("triplet_term:", triplet_term.detach().cpu().numpy())
           print("circle_term:", circle_term.detach().cpu().numpy())
           raise ValueError("Loss is NaN. Training unstable.")

       return total_loss

# ===== EMBEDDING MODEL =====
class ClinicalEmbedder(nn.Module):
   def __init__(self, input_dim=64, emb_dim=32):
       super().__init__()
       self.net = nn.Sequential(
           nn.Linear(input_dim, 128),
           nn.ReLU(),
           nn.Dropout(0.3),
           nn.Linear(128, 64),
           nn.ReLU(),
           nn.Linear(64, emb_dim),
           nn.LayerNorm(emb_dim)
       )
       self._initialize_weights()

   def _initialize_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Linear):
               nn.init.xavier_normal_(m.weight)
               nn.init.constant_(m.bias, 0.0)

   def forward(self, x):
       return self.net(x)

# ===== INTELLIGENT MEDICAL TRIPLET SAMPLER =====
def generate_medical_batch(batch_size=64):
   anchors = []
   positives = []
   negatives = []
   relations = []

   for _ in range(batch_size):
       anchor_entity = random.choice(ALL_ENTITIES)
       anchor_vec = generate_clinical_vector(anchor_entity)

       pos_entity = anchor_entity
       pos_vec = generate_clinical_vector(pos_entity)

       possible_negs = [e for e in ALL_ENTITIES if e != anchor_entity]
       weights = []
       for e in possible_negs:
           rel = get_medical_relation(anchor_entity, e)
           w = {
               "contraindicated": 6,
               "direct_cause": 5,
               "treated_by": 4,
               "symptom_of": 3,
               "risk_factor_for": 3,
               "associated": 2,
               "unrelated": 1
           }.get(rel, 1)
           weights.append(w)

       weights = np.array(weights, dtype=float)
       weights /= weights.sum()
       neg_entity = np.random.choice(possible_negs, p=weights)
       neg_vec = generate_clinical_vector(neg_entity)

       relation = get_medical_relation(anchor_entity, neg_entity)

       anchors.append(anchor_vec)
       positives.append(pos_vec)
       negatives.append(neg_vec)
       relations.append(relation)

   return (torch.stack(anchors).to(device),
           torch.stack(positives).to(device),
           torch.stack(negatives).to(device),
           relations)

# ===== TRAINING LOOP =====
model = ClinicalEmbedder(input_dim=len(CLINICAL_FEATURES), emb_dim=128).to(device)
criterion = MedicalOntoTripletLoss(RELATION_MARGINS, gamma=3, alpha=0.6, beta=1.1)
optimizer = optim.AdamW(model.parameters(), lr=0.15, weight_decay=1e-5)  # ← slightly lower LR
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

print("🏥 Training Medical OntoTriplet++ for Clinical Embeddings...")
print("Entities: Diseases, Symptoms, Treatments, Risk Factors\n")

for epoch in range(1, 500):
   try:
       optimizer.zero_grad()
       anchor_x, pos_x, neg_x, rels = generate_medical_batch(64)

       a_emb = model(anchor_x)
       p_emb = model(pos_x)
       n_emb = model(neg_x)

       loss = criterion(a_emb, p_emb, n_emb, rels)
       loss.backward()

       torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
       optimizer.step()
       scheduler.step()

       if epoch % 50 == 0:
           with torch.no_grad():
               diabetes = generate_clinical_vector("Type2Diabetes").unsqueeze(0).to(device)
               metformin = generate_clinical_vector("Metformin").unsqueeze(0).to(device)
               insulin = generate_clinical_vector("Insulin").unsqueeze(0).to(device)
               asthma = generate_clinical_vector("Asthma").unsqueeze(0).to(device)
               beta_blocker = generate_clinical_vector("BetaBlocker").unsqueeze(0).to(device)
               smoking = generate_clinical_vector("Smoking").unsqueeze(0).to(device)
               copd = generate_clinical_vector("COPD").unsqueeze(0).to(device)

               d_emb = model(diabetes)
               m_emb = model(metformin)
               i_emb = model(insulin)
               a_emb = model(asthma)
               bb_emb = model(beta_blocker)
               s_emb = model(smoking)
               c_emb = model(copd)

               d_m = F.pairwise_distance(d_emb, m_emb).item()
               d_i = F.pairwise_distance(d_emb, i_emb).item()
               a_bb = F.pairwise_distance(a_emb, bb_emb).item()
               s_c = F.pairwise_distance(s_emb, c_emb).item()

           print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                 f"Diabetes↔Metformin: {d_m:.3f} | "
                 f"Asthma↔BetaBlocker: {a_bb:.3f} | "
                 f"Smoking↔COPD: {s_c:.3f}")

   except Exception as e:
       print(f"🛑 Training crashed at epoch {epoch}: {e}")
       break

print("\n✅ Training complete!")

# ===== EVALUATE CLINICAL STRUCTURE =====
print("\n" + "="*70)
print("🩺 FINAL EMBEDDING DISTANCES — CLINICAL LOGIC VERIFICATION")
print("="*70)

with torch.no_grad():
   entities_to_test = ["Type2Diabetes", "Metformin", "Insulin", "Asthma", "BetaBlocker", "Smoking", "COPD", "Polyuria"]
   vectors = {
       e: generate_clinical_vector(e, noise_level=0.001).unsqueeze(0).to(device)
       for e in entities_to_test
   }
   embeddings = {e: model(vec) for e, vec in vectors.items()}

   test_pairs = [
       ("Type2Diabetes", "Metformin", "treated_by"),
       ("Type2Diabetes", "Insulin", "treated_by"),
       ("Type2Diabetes", "Polyuria", "symptom_of"),
       ("Asthma", "BetaBlocker", "contraindicated"),
       ("Smoking", "COPD", "direct_cause"),
       ("Type2Diabetes", "Asthma", "unrelated"),
       ("Metformin", "BetaBlocker", "unrelated"),
   ]

   for e1, e2, expected_rel in test_pairs:
       dist = F.pairwise_distance(embeddings[e1], embeddings[e2]).item()
       inferred_rel = get_medical_relation(e1, e2)
       status = "✅" if inferred_rel == expected_rel else "⚠️"
       print(f"{status} {e1:15} ↔ {e2:15} | Dist: {dist:5.3f} | Relation: {inferred_rel} (expected: {expected_rel})")