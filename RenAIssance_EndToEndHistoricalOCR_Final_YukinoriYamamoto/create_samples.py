#!/usr/bin/env python3
"""
テスト用サンプル画像生成スクリプト
文書画像セグメンテーションアプリのテスト用に簡単な文書画像を生成します。
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_sample_document(width=800, height=600, filename="sample_document.png"):
    """サンプル文書画像を作成"""

    # 白い背景の画像を作成
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # デフォルトフォントを使用（システムによって異なる）
    try:
        # Windowsの場合
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_medium = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        # フォントが見つからない場合はデフォルト
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # タイトル
    title_text = "文書セグメンテーションテスト"
    draw.text((50, 30), title_text, fill="black", font=font_large)

    # 段落1
    paragraph1 = [
        "これはテスト用の文書です。",
        "複数の行にわたって文字が",
        "記述されています。",
    ]

    y_pos = 100
    for line in paragraph1:
        draw.text((50, y_pos), line, fill="black", font=font_medium)
        y_pos += 30

    # 段落2（少し右にずらす）
    paragraph2 = ["第二段落です。", "異なる位置に配置された", "テキストブロックです。"]

    y_pos = 250
    for line in paragraph2:
        draw.text((300, y_pos), line, fill="black", font=font_medium)
        y_pos += 30

    # 段落3（小さいフォント）
    paragraph3 = [
        "小さなフォントで書かれた注釈です。",
        "このような小さな文字も検出できるかテストします。",
        "複数行にわたる注釈テキスト。",
    ]

    y_pos = 400
    for line in paragraph3:
        draw.text((50, y_pos), line, fill="gray", font=font_small)
        y_pos += 20

    # 番号付きリスト
    list_items = ["1. 最初の項目", "2. 二番目の項目", "3. 三番目の項目"]

    y_pos = 500
    for item in list_items:
        draw.text((400, y_pos), item, fill="black", font=font_medium)
        y_pos += 25

    # 画像を保存
    img.save(filename)
    print(f"サンプル画像を作成しました: {filename}")

    return filename


def create_noisy_document(width=800, height=600, filename="noisy_document.png"):
    """ノイズ入りサンプル文書画像を作成"""

    # まず通常の文書を作成
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # テキストを追加
    texts = [
        "ノイズありテスト文書",
        "この文書にはノイズが含まれています",
        "セグメンテーション精度をテストします",
        "様々な角度の文字列",
        "汚れた背景の文字",
    ]

    y_positions = [50, 150, 250, 350, 450]
    x_positions = [50, 100, 75, 120, 60]

    for i, (text, x, y) in enumerate(zip(texts, x_positions, y_positions)):
        draw.text((x, y), text, fill="black", font=font)

    # PIL画像をOpenCV形式に変換
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # ノイズを追加
    noise = np.random.normal(0, 15, cv_img.shape).astype(np.uint8)
    cv_img = cv2.add(cv_img, noise)

    # 若干のブラー
    cv_img = cv2.GaussianBlur(cv_img, (3, 3), 0)

    # 画像を保存
    cv2.imwrite(filename, cv_img)
    print(f"ノイズありサンプル画像を作成しました: {filename}")

    return filename


def create_rotated_document(width=800, height=600, filename="rotated_document.png"):
    """回転したサンプル文書画像を作成"""

    # 通常の文書を作成
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    # テキストを追加
    draw.text((100, 200), "回転テスト文書", fill="black", font=font)
    draw.text((100, 250), "この文書は5度回転しています", fill="black", font=font)
    draw.text((100, 300), "回転補正機能をテストできます", fill="black", font=font)

    # PIL画像をOpenCV形式に変換
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 5度回転
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 5, 1.0)
    cv_img = cv2.warpAffine(
        cv_img, rotation_matrix, (width, height), borderValue=(255, 255, 255)
    )

    # 画像を保存
    cv2.imwrite(filename, cv_img)
    print(f"回転サンプル画像を作成しました: {filename}")

    return filename


def main():
    """メイン関数"""
    print("テスト用サンプル画像を生成中...")

    # サンプルディレクトリを作成
    os.makedirs("samples", exist_ok=True)

    # 各種サンプル画像を作成
    create_sample_document(filename="samples/sample_document.png")
    create_noisy_document(filename="samples/noisy_document.png")
    create_rotated_document(filename="samples/rotated_document.png")

    print("\n生成されたサンプル画像:")
    print("- samples/sample_document.png: 基本的な文書")
    print("- samples/noisy_document.png: ノイズ入り文書")
    print("- samples/rotated_document.png: 回転した文書")
    print("\nこれらの画像を使ってアプリケーションをテストできます。")


if __name__ == "__main__":
    main()
