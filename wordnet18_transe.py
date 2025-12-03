# wordnet18_transe.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import WordNet18
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# 1. Dataset 분석
# -----------------------------
def analyze_wordnet18():
    dataset = WordNet18(root="data/WordNet18")

    print("=== WordNet18 데이터셋 분석 ===")
    print(f"총 split 수: {len(dataset)} (보통 train / valid / test)")

    total_triples = 0
    max_entity = -1
    max_rel = -1

    for i, data in enumerate(dataset):
        edge_index = data.edge_index  # [2, num_triplets]
        edge_type = data.edge_type    # [num_triplets]

        num_triplets = edge_index.size(1)
        num_entities_split = int(edge_index.max().item()) + 1
        num_relations_split = int(edge_type.max().item()) + 1

        total_triples += num_triplets
        max_entity = max(max_entity, num_entities_split)
        max_rel = max(max_rel, num_relations_split)

        split_name = {0: "train", 1: "valid", 2: "test"}.get(i, f"split{i}")
        print(f"[{split_name}] triplet 수: {num_triplets}")

    print(f"전체 엔티티 수(최대 index 기준): {max_entity}")
    print(f"전체 relation 수(최대 index 기준): {max_rel}")
    print(f"전체 triplet 수 (train+valid+test): {total_triples}")
    print()


# -----------------------------
# 2. helper: PYG data → triple 텐서
# -----------------------------
def build_triples_from_pyg(data) -> torch.Tensor:
    """
    data.edge_index: [2, num_triplets] -> head, tail
    data.edge_type:  [num_triplets]    -> relation
    return: [num_triplets, 3] 텐서 (h, r, t)
    """
    heads = data.edge_index[0]
    tails = data.edge_index[1]
    rels = data.edge_type
    triples = torch.stack([heads, rels, tails], dim=1)
    return triples


# -----------------------------
# 3. TransE 모델 정의
# -----------------------------
class TransE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int,
                 emb_dim: int = 100, margin: float = 1.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.margin = margin

        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)

        nn.init.xavier_uniform_(self.entity_emb.weight.data)
        nn.init.xavier_uniform_(self.relation_emb.weight.data)

    def score(self, h, r, t):
        """
        TransE score: -|| e_h + e_r - e_t ||_1
        """
        h_e = self.entity_emb(h)
        r_e = self.relation_emb(r)
        t_e = self.entity_emb(t)

        return -torch.norm(h_e + r_e - t_e, p=1, dim=-1)

    def forward(self, positive_triples, negative_triples):
        """
        positive_triples, negative_triples: [batch_size, 3]
        각 행: (h, r, t)
        """
        pos_score = self.score(
            positive_triples[:, 0],
            positive_triples[:, 1],
            positive_triples[:, 2],
        )
        neg_score = self.score(
            negative_triples[:, 0],
            negative_triples[:, 1],
            negative_triples[:, 2],
        )

        y = torch.ones_like(pos_score)
        loss = F.margin_ranking_loss(
            pos_score, neg_score, y, margin=self.margin
        )
        return loss


# -----------------------------
# 4. negative sampling
# -----------------------------
def negative_sampling(pos_triples: torch.Tensor, num_entities: int) -> torch.Tensor:
    """
    간단한 negative sampling:
    - tail 을 랜덤 엔티티로 바꿔치기
    pos_triples: [batch, 3] (h, r, t)
    """
    batch_size = pos_triples.size(0)
    neg_triples = pos_triples.clone()

    # 무작위 엔티티로 tail 교체
    random_tails = torch.randint(
        low=0, high=num_entities, size=(batch_size,), dtype=torch.long
    )
    neg_triples[:, 2] = random_tails
    return neg_triples


# -----------------------------
# 5. 학습 루프
# -----------------------------
def train_transe(num_epochs: int = 50, batch_size: int = 1024, emb_dim: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용 device:", device)

    dataset = WordNet18(root="data/WordNet18")
    train_data = dataset[0]  # 보통 0: train, 1: valid, 2: test

    triples = build_triples_from_pyg(train_data)
    num_entities = int(train_data.edge_index.max().item()) + 1
    num_relations = int(train_data.edge_type.max().item()) + 1

    print("=== TransE 학습 설정 ===")
    print(f"엔티티 수: {num_entities}")
    print(f"관계 수: {num_relations}")
    print(f"train triplet 수: {triples.size(0)}")
    print(f"임베딩 차원: {emb_dim}")
    print()

    model = TransE(num_entities, num_relations, emb_dim=emb_dim, margin=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(
        TensorDataset(triples),
        batch_size=batch_size,
        shuffle=True,
    )

    print("=== TransE 학습 시작 ===")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for (pos_triples,) in train_loader:
            pos_triples = pos_triples.to(device)
            neg_triples = negative_sampling(pos_triples, num_entities).to(device)

            optimizer.zero_grad()
            loss = model(pos_triples, neg_triples)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * pos_triples.size(0)

        avg_loss = total_loss / triples.size(0)
        print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f}")

    print("학습 완료.")


# -----------------------------
# 6. 메인
# -----------------------------
def main():
    analyze_wordnet18()
    train_transe(
        num_epochs=50,
        batch_size=1024,
        emb_dim=100,
    )


if __name__ == "__main__":
    main()
