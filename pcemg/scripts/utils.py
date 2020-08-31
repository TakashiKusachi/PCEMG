""" scriptのサブライブラリ

各scriptで共通だったり、本質と関係ない処理関数群

"""

import sys
import distutils.util

def getEnvironment():
    try:
        env = get_ipython().__class__.__name__
        if env == 'ZMQInteractiveShell':
            return 'Jupyter'
        elif env == 'TerminalInteractiveShell':
            return 'IPython'
        else:
            return 'OtherShell'
    except NameError:
        return 'Interpreter'


def is_env_notebook():
    return 'ipykernel' in sys.modules

def strtobool(x: str):
    """ 文字列真偽値 to bool 変換関数
    
    文字列で書かれた真偽値(True,Y,y,False,N,n)をpythonのbool型に変換する。

    Args:
        x (str): 文字列で書かれた真偽値。strtobool参照
    
    Returns:
        bool: 変換された真偽値
    """
    return bool(distutils.util.strtobool(x))

def type_converter(x,_type):
    """ クラスの変換関数

    pythonのconfigparserが辞書で返してくれるのはいいけど、valueがstringなので、指定の型に変換する。

    Args:
        x (object or tuple of objects): 変換対象のオブジェクト。str想定
        _type (functions): 目的の型に変換するメソッド。intやfloat。boolの場合は、distutils.util.strtoboolを推奨
    
    Returns:
        tutple of any or any type: 変換後のobject。xがtupleであればtupleで、単体であれば単体で返却する。
    """

    if isinstance(x,tuple):
        ret = tuple()
        for ite in x:
            ite = type_converter(ite,_type)
            ret = ret + (ite,)
        return ret

    if not isinstance(x,str):
        return x
    else:
        return _type(x)
