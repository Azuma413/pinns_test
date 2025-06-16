import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from simple_test import TuringPattern
import time
from tqdm import tqdm

# GPUが利用可能かチェック
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class ReactionDiffusionPINN(nn.Module):
    """
    反応拡散方程式(Gray-Scott)を学習するPINNsモデル
    
    入力: [t, x, y, du, dv, feed, kill] (7次元)
    出力: [U, V] (2次元)
    """
    
    def __init__(self, n_input=7, n_output=2, n_neurons=128, n_layers=4):
        super(ReactionDiffusionPINN, self).__init__()
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        
        # ネットワーク構築
        layers = []
        layers.append(nn.Linear(n_input, n_neurons))
        layers.append(nn.Tanh())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(n_neurons, n_output))
        
        self.network = nn.Sequential(*layers)
        
        # Xavierの初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """ネットワークの重みを初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs):
        """
        順伝播
        inputs: [t, x, y, du, dv, feed, kill]
        """
        return self.network(inputs)
    
    def physics_loss(self, inputs, outputs):
        """
        物理法則に基づく損失関数
        Gray-Scott方程式:
        ∂U/∂t = du * ∇²U - UV² + feed(1-U)
        ∂V/∂t = dv * ∇²V + UV² - (feed+kill)V
        """
        t, x, y, du, dv, feed, kill = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], \
                                      inputs[:, 3:4], inputs[:, 4:5], inputs[:, 5:6], inputs[:, 6:7]
        
        U, V = outputs[:, 0:1], outputs[:, 1:2]
        
        # 自動微分の安全な計算
        def safe_grad(outputs, inputs, create_graph=True):
            """安全な勾配計算"""
            try:
                grad = torch.autograd.grad(
                    outputs, inputs, 
                    grad_outputs=torch.ones_like(outputs),
                    create_graph=create_graph, 
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if grad is None:
                    return torch.zeros_like(inputs)
                return grad
            except:
                return torch.zeros_like(inputs)
        
        # 時間微分の計算
        dU_dt = safe_grad(U.sum(), t)
        dV_dt = safe_grad(V.sum(), t)
        
        # 空間一階微分の計算
        dU_dx = safe_grad(U.sum(), x)
        dU_dy = safe_grad(U.sum(), y)
        dV_dx = safe_grad(V.sum(), x)
        dV_dy = safe_grad(V.sum(), y)
        
        # 空間二階微分の計算
        dU_dxx = safe_grad(dU_dx.sum(), x)
        dU_dyy = safe_grad(dU_dy.sum(), y)
        dV_dxx = safe_grad(dV_dx.sum(), x)
        dV_dyy = safe_grad(dV_dy.sum(), y)
        
        # ラプラシアンの計算
        laplacian_U = dU_dxx + dU_dyy
        laplacian_V = dV_dxx + dV_dyy
        
        # 反応項
        reaction_UV2 = U * V * V
        
        # Gray-Scott方程式の残差
        pde_U = dU_dt - du * laplacian_U + reaction_UV2 - feed * (1.0 - U)
        pde_V = dV_dt - dv * laplacian_V - reaction_UV2 + (feed + kill) * V
        
        return pde_U, pde_V

def generate_training_data(n_params=5, n_time_steps=500, n_spatial_points=25, 
                          width=50, height=50, random_seed=42):
    """
    TuringPatternクラスを使用して訓練データを生成
    
    Args:
        n_params: パラメータセットの数
        n_time_steps: 時間ステップ数
        n_spatial_points: 各次元の空間サンプリング点数
        width, height: シミュレーション領域のサイズ
        random_seed: ランダムシードの設定
    
    Returns:
        inputs: [t, x, y, du, dv, feed, kill]
        targets: [U, V]
        initial_data: 初期条件データ
    """
    print("訓練データ生成中...")
    
    # ランダムシードを設定
    np.random.seed(random_seed)
    
    # simple_test.pyのスライダー範囲を参考にランダムパラメータセットを生成
    # du: 0.01 ~ 0.3, dv: 0.01 ~ 0.3, feed: 0.0 ~ 0.1, kill: 0.0 ~ 0.1
    param_sets = []
    for i in range(n_params):
        param_set = {
            "du": np.random.uniform(0.01, 0.3),
            "dv": np.random.uniform(0.01, 0.3),
            "feed": np.random.uniform(0.0, 0.1),
            "kill": np.random.uniform(0.0, 0.1)
        }
        param_sets.append(param_set)
        print(f"パラメータセット {i+1}: du={param_set['du']:.3f}, dv={param_set['dv']:.3f}, feed={param_set['feed']:.3f}, kill={param_set['kill']:.3f}")

    all_inputs = []
    all_targets = []
    initial_inputs = []
    initial_targets = []
    
    # 空間座標の生成
    x_coords = np.linspace(0, width-1, n_spatial_points)
    y_coords = np.linspace(0, height-1, n_spatial_points)
    X, Y = np.meshgrid(x_coords, y_coords)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    for i, params in enumerate(param_sets[:n_params]):
        print(f"パラメータセット {i+1}/{n_params}: {params}")
        
        # TuringPatternインスタンスを作成
        pattern = TuringPattern(width=width, height=height, **params)
        
        # 初期状態を記録
        t = 0
        U_initial = pattern.U.cpu().numpy()
        V_initial = pattern.V.cpu().numpy()
        
        # 初期条件データを追加
        for j in range(len(X_flat)):
            x_idx = int(X_flat[j])
            y_idx = int(Y_flat[j])
            
            input_vec = [t, X_flat[j], Y_flat[j], params["du"], params["dv"], 
                        params["feed"], params["kill"]]
            target_vec = [U_initial[y_idx, x_idx], V_initial[y_idx, x_idx]]
            
            initial_inputs.append(input_vec)
            initial_targets.append(target_vec)
        
        # 時間発展シミュレーション
        for t_step in range(1, n_time_steps + 1):
            pattern.update(steps=1)
            
            U_current = pattern.U.cpu().numpy()
            V_current = pattern.V.cpu().numpy()
            
            # 各空間点でのデータを記録
            for j in range(len(X_flat)):
                x_idx = int(X_flat[j])
                y_idx = int(Y_flat[j])
                
                input_vec = [t_step, X_flat[j], Y_flat[j], params["du"], params["dv"], 
                            params["feed"], params["kill"]]
                target_vec = [U_current[y_idx, x_idx], V_current[y_idx, x_idx]]
                
                all_inputs.append(input_vec)
                all_targets.append(target_vec)
    
    # numpy配列に変換してTensorに変換
    inputs_tensor = torch.FloatTensor(all_inputs).to(DEVICE)
    targets_tensor = torch.FloatTensor(all_targets).to(DEVICE)
    initial_inputs_tensor = torch.FloatTensor(initial_inputs).to(DEVICE)
    initial_targets_tensor = torch.FloatTensor(initial_targets).to(DEVICE)
    
    print(f"生成されたデータ: {inputs_tensor.shape[0]} サンプル")
    print(f"初期条件データ: {initial_inputs_tensor.shape[0]} サンプル")
    
    return inputs_tensor, targets_tensor, initial_inputs_tensor, initial_targets_tensor

def train_pinn(model, inputs, targets, initial_inputs, initial_targets, 
               epochs=100, lr=1e-3, physics_weight=1e-3):
    """
    PINNsモデルの訓練
    """
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 勾配計算のため、入力に対してrequires_gradを設定
    inputs.requires_grad_(True)
    initial_inputs.requires_grad_(True)
    
    loss_history = []
    
    print("訓練開始...")
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        
        # データ損失（初期条件）
        initial_pred = model(initial_inputs)
        data_loss = criterion(initial_pred, initial_targets)
        
        # 物理法則損失
        physics_pred = model(inputs)
        pde_U, pde_V = model.physics_loss(inputs, physics_pred)
        physics_loss = criterion(pde_U, torch.zeros_like(pde_U)) + \
                      criterion(pde_V, torch.zeros_like(pde_V))
        
        # 全体の損失
        total_loss = data_loss + physics_weight * physics_loss
        
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}, "
                  f"Data Loss = {data_loss.item():.6f}, "
                  f"Physics Loss = {physics_loss.item():.6f}")
    
    return loss_history

def visualize_prediction(model, params, time_steps=[0, 100, 200, 300, 400, 500],
                        width=50, height=50):
    """
    学習済みモデルの予測を可視化（正解データと比較）
    上段：TuringPatternクラスを用いた正解データ
    下段：学習済みモデルの予測データ
    """
    model.eval()
    fig, axes = plt.subplots(2, len(time_steps), figsize=(20, 8))

    # TuringPatternインスタンスを作成して正解データを生成
    print("正解データ生成中...")
    pattern = TuringPattern(width=width, height=height, **params)
    
    # 正解データの保存用リスト
    true_data = {}
    
    # 初期状態を記録
    true_data[0] = {
        'U': pattern.U.cpu().numpy().copy(),
        'V': pattern.V.cpu().numpy().copy()
    }
    
    # 指定された時間ステップまでシミュレーションを実行
    max_time = max(time_steps)
    current_step = 0
    
    for target_step in range(1, max_time + 1):
        pattern.update(steps=1)
        current_step += 1
        
        # 指定された時間ステップのデータを保存
        if current_step in time_steps:
            true_data[current_step] = {
                'U': pattern.U.cpu().numpy().copy(),
                'V': pattern.V.cpu().numpy().copy()
            }
    
    print("予測データ生成中...")
    
    with torch.no_grad():
        for i, t in enumerate(time_steps):
            # 学習モデルによる予測用の入力を準備
            inputs = []
            for y in range(height):
                for x in range(width):
                    inputs.append([t, x, y, params["du"], params["dv"], 
                                 params["feed"], params["kill"]])
            
            inputs_tensor = torch.FloatTensor(inputs).to(DEVICE)
            predictions = model(inputs_tensor).cpu().numpy()
            
            # U成分の予測値を分離
            U_pred = predictions[:, 0].reshape(height, width)
            
            # 正解データを取得
            U_true = true_data[t]['U']
            
            # 上段：正解データ（TuringPattern）
            im1 = axes[0, i].imshow(U_true, cmap='viridis', vmin=0, vmax=1)
            axes[0, i].set_title(f'正解データ (t={t}step)', fontsize=12)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            # 下段：予測データ（学習済みモデル）
            im2 = axes[1, i].imshow(U_pred, cmap='viridis', vmin=0, vmax=1)
            axes[1, i].set_title(f'予測データ (t={t}step)', fontsize=12)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
    
    # カラーバーを追加
    plt.colorbar(im1, ax=axes[0, :], shrink=0.8, aspect=20)
    plt.colorbar(im2, ax=axes[1, :], shrink=0.8, aspect=20)
    
    plt.suptitle('反応拡散方程式の予測結果比較 (U成分)', fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    """メイン実行関数"""
    # 訓練データの生成
    inputs, targets, initial_inputs, initial_targets = generate_training_data(
        n_params=100, n_time_steps=500, n_spatial_points=20, width=50, height=50
    )
    
    # モデルの作成
    model = ReactionDiffusionPINN(n_input=7, n_output=2, n_neurons=128, n_layers=4)
    
    # 訓練
    loss_history = train_pinn(
        model, inputs, targets, initial_inputs, initial_targets,
        epochs=100, lr=1e-3, physics_weight=1e-4
    )
    
    # 損失の可視化
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('訓練損失の推移')
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    # 予測結果の可視化
    test_params = {"du": 0.14, "dv": 0.06, "feed": 0.035, "kill": 0.058}
    visualize_prediction(model, test_params, time_steps=[0, 100, 200, 300, 400, 500],)
    
    print("訓練完了！")
    
    return model

if __name__ == "__main__":
    model = main()
    # モデルを保存
    torch.save(model.state_dict(), 'reaction_diffusion_pinn.pth')
