from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import glob
import os
import torch
import time
from pydub import AudioSegment

# 学習済みモデルが保存されているパス
model_path = "./models100+"

# デバイスの選択
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# モデルとプロセッサの読み込み
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
processor = WhisperProcessor.from_pretrained(model_path)

# 処理の開始時刻を記録
start_time = time.time()

# 音声ファイルを30秒ごとに分割する関数
def split_audio(file_path, segment_length=30):
    audio = AudioSegment.from_file(file_path)
    segments = []
    
    for start in range(0, len(audio), segment_length * 1000):
        end = min(start + segment_length * 1000, len(audio))
        segment = audio[start:end]
        segments.append(segment)
    
    return segments

# 処理するMP3ファイルのリストを作成
audio_files = glob.glob("./*.mp3")  # 同じディレクトリにあるすべてのMP3ファイルを取得

# ./テキスト/ ディレクトリがなければ作成
os.makedirs("./テキスト", exist_ok=True)

# 各音声ファイルを処理
for input_audio_file in audio_files:
    print(f"Processing: {input_audio_file}")
    segments = split_audio(input_audio_file)

    # 文字起こし結果を格納するリスト
    full_transcription = []

    # 分割したセグメントを処理
    for i, segment in enumerate(segments):
        # セグメントを音声ファイルとして保存
        segment_file_path = f"./テキスト/temp_segment_{i}.mp3"
        segment.export(segment_file_path, format="mp3")
        
        try:
            # 音声データの読み込み
            audio_data, sampling_rate = librosa.load(segment_file_path, sr=16000)

            # 音声データの処理
            inputs = processor(audio_data, return_tensors="pt", sampling_rate=sampling_rate).to(device)

            # モデルによる生成
            outputs = model.generate(**inputs)

            # 結果のデコード
            transcription = processor.batch_decode(outputs, skip_special_tokens=True)

            # 不要な文字の置き換え
            clean_transcription = [text.replace("['", "").replace("']", "") for text in transcription]

            # 文字起こし結果をリストに追加
            full_transcription.extend(clean_transcription)

            # セグメントファイルを削除
            os.remove(segment_file_path)

        except Exception as e:
            print(f"Error processing segment {i} of {input_audio_file}: {e}")

    # テキストファイル名の設定
    text_file_name = f"./テキスト/{os.path.basename(input_audio_file).split('.')[0]}.txt"
    
    # まとめた文字起こしデータをファイルに保存
    with open(text_file_name, "w", encoding="utf-8") as f:
        for line in full_transcription:
            f.write(line + "\n")
    
    print(f"{text_file_name} に文字起こしを保存しました。")

    # 音声ファイルを削除
    os.remove(input_audio_file)

print("すべてのファイルの処理が完了しました。")

# 処理時間を計算
end_time = time.time()
elapsed_time = end_time - start_time

# 処理時間を分と秒に変換
minutes, seconds = divmod(elapsed_time, 60)

print(f"処理にかかった時間: {int(minutes)}分 {int(seconds)}秒")