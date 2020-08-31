
前処理
==============================
.. contents:: 目次
    :depth: 2

データの取得
==============================

本研究では :numref:`dataset` を用いた。Zinqデータセットは、JT-VAEモデルの著者らの `データ <jt_vae_>`_ を利用した。MassBankデータセットはgithub上に公開されている `リポジトリ <massbank_>`_ から取得した。

データの取得から基本的なバイナリ化（文字列として保存されているデータを数値などに変換する）までの処理の例を `ここ <datafetch_>`_ に示す。

.. csv-table:: 用いたデータセット
    :header: "データ元","データ数"
    :name: dataset

    "MassBank",""
    "Zinq",""


コードの解説（Datafetch）
-----------------------------

zinqとmassbankのデータはweb上から取得した。zinqのSMILESリストはwgetを用いて取得した。MassBankデータセットは `リポジトリ <massbank_>`_ から直接クローンした。

.. code-block:: sh

    ! git clone https://github.com/MassBank/MassBank-data.git 
    ! wget https://github.com/TakashiKusachi/icml18-jtnn/raw/master/data/zinc/all.txt

MassBank-dataリポジトリは測定者ごとにフォルダが構成され、その中に一つのスペクトルごとにファイルが作成されている。ここでは、無作為に全部のfileのpathをリストとして取得した。

.. code-block:: python

    file_list = list(Path("./MassBakn-data").glob("*/*.txt"))
    print("number of file list: {}".format(len(file_list)))

スペクトルデータは測定条件や分子の情報とともに文字列の状態でfileに保存されているため、pythony上で扱いやすいように変換する必要がある（前述のバイナリ化）。ファイルのpathから情報を抽出してdictで返すプログラムを下にしめす。データによって保存されている情報が違うため、無理に全部の情報を取得しようとせず必要なモノだけ取得するようにした。

.. code-block:: python

    AUTHORS = "AUTHORS"
    AC_INSTRUMENT = "AC$INSTRUMENT"
    AC_INSTRUMENT_TYPE = "AC$INSTRUMENT_TYPE"
    CH_NAME = "CH$NAME"
    CH_SMILES = "CH$SMILES"
    AC_MASS_SPECTROETRY = "AC$MASS_SPECTROMETRY"
    MS_TYPE = "MS_TYPE"
    ION_MODE = "ION_MODE"
    IONIZATION_ENERGY = "IONIZATION_ENERGY"

    def analyze(args):
        num,path = args
        #param init 
        auth = None
        inst = None
        inst_type = None
        name = None
        smiles = None
        ms_type = None
        ion_mode = None
        ret = {}
        with open(path) as f:
            temp = f.read().split("\n")
            for num,line in enumerate(temp):
                row = line.split(": ",1)
                if row[0] == AUTHORS:
                    ret["authors"]= row[1]
                elif row[0] == AC_INSTRUMENT:
                    ret["instrument"] = row[1]
                elif row[0] == AC_INSTRUMENT_TYPE:
                    ret["instrument_type"] = row[1]
                elif row[0] == CH_NAME:
                    ret["name"] = row[1]
                elif row[0] == CH_SMILES:
                    mol = Chem.MolFromSmiles(row[1])
                    if mol is None:
                        ret["smiles"] = row[1]
                    else:
                        ret["smiles"] = Chem.MolToSmiles(mol)
                elif row[0] == AC_MASS_SPECTROETRY:
                    label,_type = row[1].split(" ",1)
                    if label == MS_TYPE:
                        ret["ms_type"] = _type
                    elif label == ION_MODE:
                        ret["ion_mode"] = _type
                    elif label == IONIZATION_ENERGY:
                        ret["ionization_energy"] = _type
                elif row[0] == "PK$PEAK":
                    peak_start = num+1
            peak_x,peak_y = zip(*[one.split(" ")[2:4] for one in temp[peak_start:-1] if one != "//"])
            ret["peak_x"] = np.array(peak_x,dtype=np.float32)
            ret["peak_y"] = np.array(peak_y,dtype=np.float32)
        #ret["path"] = path
        return ret



.. _jt_vae: https://github.com/wengong-jin/icml18-jtnn
.. _massbank: https://github.com/MassBank/MassBank-data.git
.. _datafetch: https://github.com/TakashiKusachi/PCEMG/blob/master/example/DataFetch.ipynb