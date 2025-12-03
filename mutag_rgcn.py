# mutag_rgcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool


# -----------------------------
# 1. Dataset 분석
# -----------------------------
def analyze_mutag(dataset: TUDataset) -> None:
    print("=== MUTAG 데이터셋 분석 ===")
    print(f"총 그래프 수: {len(dataset)}")
    print(f"클래스 수: {dataset.num_classes}")

    # 라벨 분포
    labels = torch.tensor([data.y.item() for data in dataset])
    label_counts = torch.bincount(labels)
    for i, c in enumerate(label_counts):
        print(f"  클래스 {i}: {c.item()}개")

    # 노드 feature 차원
    print(f"노드 feature 차원: {dataset.num_node_features}")

    # 그래프별 노드/엣지 통계
    num_nodes = torch.tensor([data.num_nodes for data in dataset], dtype=torch.float)
    num_edges = torch.tensor([data.num_edges for data in dataset], dtype=torch.float)
    print(f"평균 노드 수: {num_nodes.mean().item():.2f}")
    print(f"평균 엣지 수: {num_edges.mean().item():.2f}")

    # 엣지 feature(결합 타입) 확인
    data0 = dataset[0]
    print("--- 첫 번째 그래프 예시 ---")
    print(f"노드 수: {data0.num_nodes}")
    print(f"엣지 수: {data0.num_edges}")
    if getattr(data0, "edge_attr", None) is not None:
        print(f"edge_attr shape: {data0.edge_attr.shape}")
        edge_type0 = data0.edge_attr.argmax(dim=1)
        num_rel_types0 = int(edge_type0.max().item()) + 1
        print(f"(첫 그래프 기준) 관계 타입 수: {num_rel_types0}")
    else:
        print("edge_attr 없음 (엣지 타입 정보가 없음).")


# -----------------------------
# 2. 전체 relation 타입 개수 세기
# -----------------------------
def count_num_relations(dataset: TUDataset) -> int:
    """
    전체 그래프를 돌면서 edge_attr(one-hot) -> edge_type index 로 바꾸고
    그 중 최대값을 이용해 relation 타입 개수를 계산한다.
    """
    all_edge_types = []

    for data in dataset:
        if getattr(data, "edge_attr", None) is not None:
            edge_type = data.edge_attr.argmax(dim=1)
            all_edge_types.append(edge_type)

    if not all_edge_types:
        # edge_attr 가 전혀 없는 경우: 타입 1개라고 가정
        return 1

    cat = torch.cat(all_edge_types)
    num_relations = int(cat.max().item()) + 1
    return num_relations


# -----------------------------
# 3. R-GCN 모델 정의
# -----------------------------
class RGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_relations: int):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        # edge_attr: [num_edges, num_edge_features] (one-hot)
        edge_type = edge_attr.argmax(dim=1)  # [num_edges]

        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


# -----------------------------
# 4. 학습 & 평가 루프
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += data.num_graphs

    return correct / total


# -----------------------------
# 5. 메인: 데이터 로드 + 분석 + 학습
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용 device:", device)

    dataset = TUDataset(root="data/MUTAG", name="MUTAG")
    analyze_mutag(dataset)

    # 🔥 전체 그래프를 기준으로 relation 타입 수 계산
    num_relations = count_num_relations(dataset)
    print(f"전체 relation 타입 수: {num_relations}")

    # train / test split
    torch.manual_seed(0)
    dataset = dataset.shuffle()
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = RGCN(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        num_relations=num_relations,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    print("\n=== R-GCN 학습 시작 ===")
    for epoch in range(1, 51):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_acc = eval_accuracy(model, train_loader, device)
        test_acc = eval_accuracy(model, test_loader, device)
        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss:.4f} | "
            f"Train Acc {train_acc:.3f} | "
            f"Test Acc {test_acc:.3f}"
        )


if __name__ == "__main__":
    main()
