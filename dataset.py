import whisper
import torch
import os
import csv

# 現在利用可能なデバイスを確認
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Whisperモデルをロード
model = whisper.load_model("large-v3", device=device)

# 出力先のCSVファイルパスを定義
output_csv_file = 'train0.csv'

# 音声ファイルが格納されているディレクトリのパス
out_directory = './clips'  # あなたの音声ファイルのパスに変更してください

# サンプリングレートを定義（必要に応じて変更してください）
SAMPLING_RATE = 16000  # 例: 16kHz

# CSVファイルを開き、結果を書き込む
with open(output_csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # CSVのヘッダーを書き込む
    writer.writerow(['path', 'sampling_rate', 'correct', 'whisper'])

    # 指定されたディレクトリ内の全てのファイルをループ処理
    for filename in os.listdir(out_directory):
        if filename.endswith(".mp3"):  # ファイルが.mp3で終わる場合
            file_path = os.path.join(out_directory, filename)  # ファイルのフルパスを取得
            print(f'Processing file: {file_path}')
            result = model.transcribe(file_path, language="ja")
            # 結果をCSVに書き込む
            writer.writerow([file_path, SAMPLING_RATE, '', result["text"]])
            print(result["text"])

print(f"Processing complete. Results saved to {output_csv_file}")