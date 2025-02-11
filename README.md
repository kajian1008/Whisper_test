## Whisper_test
Whisper_Airakuの文字起こし。

### 設定
ランタイム〉ランタイムのタイプを変更\
・ランタイムのタイプ＝Python 3\
・ハードウェア アクセラレータ＝T4 GPU\

### 準備
・Googleのマイドライブ直下にWhisperモデルをアップロード\
例："/content/drive/MyDrive/models100+"\
・マイドライブに音声フォルダー（audio）を作成\
例："/content/drive/MyDrive/audio/"\
・audioフォルダーに起こしたい音声ファイルをアップロード\

### 実行
・okoshite.ipynbを実行\
ランタイム〉すべてのセルを実行\
マイドライブに「テキスト」フォルダーが作成され、文字起こしされたテキストデータが保存される。\
