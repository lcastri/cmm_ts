from models.utils import Words as W


config = {
    W.FOLDER : None,
    W.NPAST : None,
    W.NFUTURE : None,
    W.NDELAY : None,
    W.NFEATURES : None,
    W.FEATURES : None,
    W.USEATT : False,
    W.USECAUSAL : False,
    W.CTRAINABLE : None,
    W.USECONSTRAINT : False,
    W.TRAINTHRESH : None,
    W.ATTUNITS : 256,
    W.ENCDECUNITS : 256,
    W.DECINIT : False,
    W.D1UNITS : 128,
    W.D1ACT : "tanh",
    W.D2UNITS : 64,
    W.D2ACT : "tanh",
    W.D3UNITS : 32,
    W.D3ACT : "tanh",
    W.DRATE : 0.4,
}