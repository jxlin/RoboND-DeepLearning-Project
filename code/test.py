
from model.common import *
from model.models import *

_model = RFCNsimpleModel( inWidth = 160, inHeight = 160, 
                          inDepth = 3, numclasses = 3 )

# _model = RFCNdeeperModel( inWidth = 160, inHeight = 160, 
#                           inDepth = 3, numclasses = 3 )

# _model = RFCNvggModel( inWidth = 160, inHeight = 160, 
#                        inDepth = 3, numclasses = 3 )

print( _model )
_model.show()