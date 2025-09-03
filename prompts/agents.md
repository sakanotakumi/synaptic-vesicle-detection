1. 目的・成果物
	•	目的：FIB-SEM 等の3Dボリュームからシナプス小胞（SV）の中心座標とだいたいの半径、信頼度スコアを自動抽出。
	•	成果物（最低限）
	  •	sv_points.parquet / sv_points.csv：列 id, z_vox, y_vox, x_vox, z_um, y_um, x_um, r_vox, r_nm, score, tile_id
	  •	sv_points_napari.csv（Napari Points 互換：z,y,xと任意のproperties）

2. 入出力仕様

入力
	•	ボリューム：tiff スタック
	•	ボクセルサイズ（nm）：例 --voxel-size 8 8 8 (z, y, x)
	•	想定SV径（nm）：例 --diam-range 30 50
	•	（任意）マスク：細胞内/シナプス領域などの 0/1 3D マスク。検出はマスク内に限定。

出力
	•	parquet/csv（上記スキーマ）
	•	Napari Points 互換 CSV
	•	サマリ metrics.jsonl（タイルごとの件数、平均半径、スコア分布、処理時間など）
	•	QC 画像（サンプル cutout、ヒストグラム、スコア×リングネス散布図）

3. 依存関係・環境
	•	Python 3.10+
	•	ライブラリ：numpy, scipy, scikit-image, pandas, pyarrow, zarr, dask[array], tqdm, numba（高速化）, cupy（任意; GPU LoG/畳み込み）, opencv-python（cutout保存任意）
	•	大規模処理：dask による 256³〜384³ タイル分割＋ハロー（例: 16〜24 voxel）

4. 前処理
	•	強度正規化（タイル単位）：
	•	2–98% 分位でクリップ → z-score or min-max（オプション）

5. 候補生成（LoG/DoG：要件）
	•	スケール空間：SV半径 r と Gaussian σ の関係は r \approx \sqrt{3}\,\sigma
	•	与えた径範囲 [d_min, d_max]（nm）から半径範囲 [r_min, r_max] = [d_min/2, d_max/2]
	•	ボクセル変換：r_vox = r_nm / (voxel_size_nm)（軸ごと）
	•	σ候補列：σ_k = r_vox_k / sqrt(3) を 4–6 段階で均等サンプリング
	•	応答：スケール正規化 LoG
	•	R_k = s_k^2 * (− gaussian_laplace(I, sigma=σ_k))（暗環/明中心前提で正値化）
	•	代替：DoG でも可（G(σ)差分×スケール正規化）
	•	極大検出：
	•	R_max = max_k R_k を取り、peak_local_max（3×3×3 近傍）で3D極大抽出
	•	閾値は パーセンタイル（例：R_max の 99.5%）か Top-K/体積（例：100 個/10⁶voxel）
	•	スケール推定：各ピークの k* = argmax_k R_k から σ_hat、r_hat_vox = sqrt(3) * σ_hat を記録

6. NMS（重複排除）
	•	半径依存 NMS：候補を score = R_max[p] で降順に並べ、
近傍距離 < α * r_hat_vox（例 α=0.8〜1.2）内の低スコア点を抑圧
	•	タイル境界：隣接タイル間で再NMS（同条件）

7. 特徴量計算と絞り込み（方針 2）

7.1 リングネス（“暗い殻＋明るい中心”）
	•	定義：ringness = mean(annulus) − mean(inner)
	•	球状内域：半径 [0, r_in]（例 r_in = r_hat_vox − 2）
	•	球状外環：半径 [r_out−Δ, r_out]（例 r_out = r_hat_vox, Δ=2）
	•	SV想定：ringness < −τ_r（負の大きいほど良い）
	•	実装：事前に球マスク/殻マスクの座標オフセットを離散生成し、np.add.reduceat か numba で高速平均
	•	複数半径でロバスト化：r_in/out を ±1 voxel で 3 通り評価し最小（最悪）ringness を採用

7.2 球状度（Hessian ベース）
	•	平滑：gaussian(I, sigma=σ_hat)
	•	Hessian を各候補点近傍でサンプリングし固有値 |λ1|≥|λ2|≥|λ3|（絶対値で整列）
	•	等方性指標：isotropy = |λ3| / |λ1|（0〜1; 1 に近いほど球状）
	•	負曲率性：neg = sign(λ1 + λ2 + λ3) が想定と矛盾しないこと（オプション）
	•	受理条件：isotropy ≥ τ_iso（例 0.6〜0.8） かつ |λ1| ≥ τ_hess

7.3 スコア統合と最終判定
	•	score = w1 * z(R_max) + w2 * z(−ringness) + w3 * z(isotropy)（z はタイル内 z-score）
	•	閾値 score ≥ τ_score または 上位 top-K/体積 で最終採択
	•	各 τ と w は YAML/CLI で設定可能（デフォルトは自動推奨値）

8. パラメータ自動化（デフォルト設計）
	•	σ_list：d_min, d_max, voxel_size から 4〜6本を自動生成
	•	nms_radius = clamp(round(0.9 * r_hat_vox), 3, 10)
	•	ringness：τ_r はタイル内 ringness 分布の 下位 p 分位（例 20%）を初期値に
	•	τ_iso, τ_hess：Hessian 指標分布の右側モードから自動初期化（Otsu など）

9. スケーラビリティ・性能要件
	•	タイル：デフォルト 256³（ハロー 16〜24）可変
	•	並列：タイル単位で並列（multiprocessing or Dask distributed）
	•	GPU（任意）：cupy があれば Gaussian/LoG を GPU 実装に自動切換え

10. ロギング／再現性
	•	乱数は固定, seed=47
	•	すべての実行設定と派生パラメータ（σ列、閾値、NMS α 等）を run_config.yaml に保存
	•	進捗＆統計を metrics.jsonl に逐次追記（タイル単位）
