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
    W.ATTUNITS : 128,
    W.ENCDECUNITS : 128,
    W.DECINIT : False,
    W.D1UNITS : 64,
    W.D1ACT : "relu",
    W.D2UNITS : 32,
    W.D2ACT : "relu",
}