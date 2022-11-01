from models.utils import Words as W


CONFIG = {
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
    W.T2VUNITS : 128,
    W.ENCDECUNITS : 128,
    W.DECINIT : False,
    W.DACT : "tanh",
    W.DRATE : 0.4,
}