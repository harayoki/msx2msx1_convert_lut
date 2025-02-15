import numpy as np
import colorsys
import argparse
from typing import Tuple, List
from pathlib import Path

# **📌 LUT 設定（調整可能）**
DEFAULT_LUMINANCE_WEIGHT = 0.70  # 輝度の影響（明るさを調整）
DEFAULT_HUE_WEIGHT = 0.75  # 色相の影響（色ズレ補正）
DEFAULT_SATURATION_WEIGHT = 0.3  # 彩度の影響（大きくしすぎると色がくすむ）
default_weights: List[float] = [DEFAULT_HUE_WEIGHT, DEFAULT_SATURATION_WEIGHT, DEFAULT_LUMINANCE_WEIGHT]

# **📌 MSX1 / MSX2 のカラーパレット（RGB値, 16進数表記）**
MSX1_COLORS = np.array([
        (0x00, 0x00, 0x00),  # Black (0, 0, 0)
        (0x3E, 0xB8, 0x49),  # Medium Green (62, 184, 73)
        (0x74, 0xD0, 0x7D),  # Light Green (116, 208, 125)
        (0x59, 0x55, 0xE0),  # Dark Blue (89, 85, 224)
        (0x80, 0x76, 0xF1),  # Light Blue (128, 118, 241)
        (0xB9, 0x5E, 0x51),  # Dark Red (185, 94, 81)
        (0x65, 0xDB, 0xEF),  # Cyan (101, 219, 239)
        (0xDB, 0x65, 0x59),  # Medium Red (219, 101, 89)
        (0xFF, 0x89, 0x7D),  # Light Red (255, 137, 125)
        (0xCC, 0xC3, 0x5E),  # Dark Yellow (204, 195, 94)
        (0xDE, 0xD0, 0x87),  # Light Yellow (222, 208, 135)
        (0x3A, 0xA2, 0x41),  # Dark Green (58, 162, 65)
        (0xB7, 0x66, 0xB5),  # Magenta (183, 102, 181)
        (0xCC, 0xCC, 0xCC),  # Gray (204, 204, 204)
        (0xFF, 0xFF, 0xFF),   # White (255, 255, 255)
])
MSX2_COLORS = np.array([
        (0x00, 0x00, 0x00),  # Black (0, 0, 0)
        (0x22, 0xDD, 0x22),  # Medium Green (34, 221, 34)
        (0x66, 0xFF, 0x66),  # Light Green (102, 255, 102)
        (0x22, 0x22, 0xFF),  # Dark Blue (34, 34, 255)
        (0x44, 0x66, 0xFF),  # Light Blue (68, 102, 255)
        (0xAA, 0x22, 0x22),  # Dark Red (170, 34, 34)
        (0x44, 0xDD, 0xFF),  # Cyan (68, 221, 255)
        (0xFF, 0x22, 0x22),  # Medium Red (255, 34, 34)
        (0xFF, 0x66, 0x66),  # Light Red (255, 102, 102)
        (0xDD, 0xDD, 0x22),  # Dark Yellow (221, 221, 34)
        (0xDD, 0xDD, 0x88),  # Light Yellow (221, 221, 136)
        (0x22, 0x88, 0x22),  # Dark Green (34, 136, 34)
        (0xDD, 0x44, 0xAA),  # Magenta (221, 68, 170)
        (0xAA, 0xAA, 0xAA),  # Gray (170, 170, 170)
        (0xFF, 0xFF, 0xFF),   # White (255, 255, 255)
])

# 入力の色味を微調整したいときに使う
MSX2_TO_MSX1_ADJUSTED = np.array([
    (0, 0, 0),  # Black
    (0, 0, 0),  # Medium Green
    (0, 0, 0),  # Light Green
    (0, 0, 0),  # Dark Blue
    (0, 0, 0),  # Light Blue
    (0, 0, 0),  # Dark Red
    (0, 0, 0),  # Cyan
    (0, 0, 0),  # Medium Red
    (0, 0, 0),  # Light Red
    (0, 0, 0),  # Dark Yellow
    (0, 0, 0),  # Light Yellow
    (0, 0, 0),  # Dark Green
    (0, 0, 0),  # Magenta
    (0, 0, 0),  # Gray
    (0, 0, 0),  # White
])

MSX1_TO_MSX2_ADJUSTED = np.array([
    (0, 0, 0),  # Black
    (0, 0, 0),  # Medium Green
    (0, 0, 0),  # Light Green
    (0, 0, 0),  # Dark Blue
    (0, 0, 0),  # Light Blue
    (0, 0, 0),  # Dark Red
    (0, 0, 0),  # Cyan
    (0, 0, 0),  # Medium Red
    (0, 0, 0),  # Light Red
    (0, 0, 0),  # Dark Yellow
    (0, 0, 0),  # Light Yellow
    (0, 0, 0),  # Dark Green
    (0, 0, 0),  # Magenta
    (0, 0, 0),  # Gray
    (0, 0, 0),   # White
])


# **📌 RGB を HSV に変換**
def rgb_to_hsv_adjusted(rgb):
    h, s, v = colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    return np.array([h, s, v])


def find_best_match_color(input_rgb, reference_colors, hsl_weight: List[float]) -> int:
    input_hsl = rgb_to_hsv_adjusted(input_rgb)

    # **H, S, V の各軸の重みを適用**
    input_vector = np.array([
        input_hsl[0] * hsl_weight[0],
        input_hsl[1] * hsl_weight[1],
        input_hsl[2] * hsl_weight[2]
    ])

    best_index = None
    best_distance = float('inf')

    distances = []
    for idx, ref_rgb in enumerate(reference_colors):
        ref_hsl = rgb_to_hsv_adjusted(ref_rgb)

        # **参照色も同じ重みでベクトル化**
        ref_vector = np.array([
            ref_hsl[0] * hsl_weight[0],
            ref_hsl[1] * hsl_weight[1],
            ref_hsl[2] * hsl_weight[2]
        ])

        # **ユークリッド距離で最近傍を決定**
        hsl_distance = np.linalg.norm(input_vector - ref_vector)

        # RGBの距離も計算
        input_rgb_temp = input_rgb.copy() / 255
        # input_rgb_temp[0] *= 1
        # input_rgb_temp[1] *= 1
        # input_rgb_temp[2] *= 1
        ref_rgb_temp = ref_rgb.copy() / 255
        # ref_rgb_temp[0] *= 1
        # ref_rgb_temp[1] *= 1
        # ref_rgb_temp[2] *= 1
        rgb_distance = np.linalg.norm(input_rgb_temp - ref_rgb_temp)

        total_distance = np.linalg.norm([hsl_distance * 1, rgb_distance * 1])

        distances.append([idx, total_distance])

    # **距離の近い順にソート**
    distances.sort(key=lambda x: x[1])

    idx1, d1 = distances[0]

    # # **距離に応じたブレンド割合を計算（近い方を優先）**
    # idx2, d2 = distances[1]
    # total_dist = d1 + d2
    # if total_dist == 0:  # 完全一致
    #     ratio1, ratio2 = 1.0, 0.0
    # else:
    #     ratio1 = d2 / total_dist  # 近い方の割合を大きく
    #     ratio2 = d1 / total_dist  # 遠い方の割合を小さく
    #     ratio1 *= 1.0
    #     ratio2 *= 0.0
    #     ratio1 = ratio1 / (ratio1 + ratio2)
    #     ratio2 = ratio2 / (ratio1 + ratio2)
    #
    # return idx1, idx2, ratio1, ratio2

    return idx1


def main(palette_from: np.ndarray, palette_to: np.ndarray, color_adjustment: np.ndarray | None,
         output_path: Path, hsl_weight: List[float], lut_size: int = 8) -> None:
    # **📌 3D LUTの生成**
    lut_3d_data = []
    palette_from_adjusted = palette_from
    if color_adjustment is not None:
        palette_from_adjusted = palette_from + color_adjustment
    for b in range(lut_size):
        for g in range(lut_size):
            for r in range(lut_size):
                # **RGB値を正規化**
                input_rgb = np.array([r, g, b]) / (lut_size - 1) * 255
                # **最適なカラーを選択**
                idx1 = find_best_match_color(input_rgb, palette_from_adjusted, hsl_weight)
                rgb = palette_to[idx1] / 255.0
                # R G B の順番で保存
                lut_3d_data.append(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}")

    # **📌 LUT ファイルを作成**
    with open(output_path, "w") as f:
        f.write(f"TITLE \"MSX2 to MSX1 3D LUT (HSL Vector, {lut_size}x{lut_size}x{lut_size})\"\n")
        f.write(f"LUT_3D_SIZE {lut_size}\n")
        f.write("\n".join(lut_3d_data))

    print(f"> {output_path.as_posix()}")


def parse_args() -> None:
    parser = argparse.ArgumentParser(description="MSX2 から MSX1 への変換用 3D LUT を作成します")
    parser.add_argument("--mode", type=str, choices=["msx2_to_msx1", "msx1_to_msx2", "2->1", "1->2"],
                        default="msx2_to_msx1", help="変換モード")
    parser.add_argument("-o", "--output", type=str, default="", help="出力ディレクトリまたはファイル名")
    parser.add_argument("--size", type=int, choices=[4, 8, 16 ,32], default=8, help="3D LUTのサイズ")
    # parser.add_argument("-hsl", "--hsl_weight", nargs=3, type=float, default=default_weights,
    #                     help=f"HSLの変換の重み（H, S, V）default: {default_weights}")
    args = parser.parse_args()
    output_path = args.output
    mode = args.mode
    if mode == "2->1":
        mode = "msx2_to_msx1"
    elif mode == "1->2":
        mode = "msx1_to_msx2"
    lut_size = args.size
    hsl_weight = [0, 0, 0]
    if output_path == "":
        output_path = Path(".")
    output_path = Path(output_path)
    if output_path.is_dir():
        output_path = output_path / f"{mode}.cube"
    if mode == "msx2_to_msx1" or mode == "2->1":
        main(MSX2_COLORS, MSX1_COLORS, MSX2_TO_MSX1_ADJUSTED, output_path, hsl_weight, lut_size)
    else:
        main(MSX1_COLORS, MSX2_COLORS, MSX1_TO_MSX2_ADJUSTED, output_path, hsl_weight, lut_size)



if __name__ == "__main__":
    parse_args()
