## ゼミの研究用レポジトリ
### 基本構成
```
tokai_teio/
  ├ README.md
  ├ config/
  │  ├ logging.yml
  │  └ service_account.json
  ├ inspections/
  │  └ {project_name}/
  │      └ README.md
  ├ main_research/
  │  └ {research_name}/
  │      └ README.md
  └ workspace/
```
+ config
  + logging の設定ファイルや、特定の Google Spreadsheet へのアクセスに必要な service_account.json を格納する
  + なお、service_account.json ファイルはバージョン管理外となるため、コピーして設置する必要がある
+ inspection
  + 新規モデルの検証や勉強するためのスクリプトを格納。トピックごとにディレクトリは分ける。
+ main_research
  + 研究用のディレクトリ
+ workspace
  + Gitで管理する必要もないスクリプトを格納する。ちょっとした検証用。

## 学習データと出力データの取り扱い
基本各プロジェクトディレクトリに `data/`, `output/` を作成しその中に格納。
また `.gitignore`を `data/`, `output/` 配下に 
```
*
!.gitignore
```
で作成しGitの管理対象外にする。

## 環境構築
プロジェクトごとに環境を作成する。
Dockerを用いない場合は基本pyenv + Pipenvを使う。
pipenvで作成される`.venv`は各プロジェクトのルートに配置する。

