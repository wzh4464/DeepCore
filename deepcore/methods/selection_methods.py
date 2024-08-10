###
 # File: /selection_methods.py
 # Created Date: Friday, August 9th 2024
 # Author: Zihan
 # -----
 # Last Modified: Saturday, 10th August 2024 5:30:53 pm
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

# selection_methods.py

from .cal import Cal
from .contextualdiversity import ContextualDiversity
from .craig import Craig
from .deepfool import DeepFool
from .forgetting import Forgetting
from .full import Full
from .glister import Glister
from .grand import GraNd
from .herding import Herding
from .kcentergreedy import kCenterGreedy
from .submodular import Submodular
from .uncertainty import Uncertainty
from .uniform import Uniform

# the list of available selection methods
SELECTION_METHODS = {
    'Cal': Cal,
    'ContextualDiversity': ContextualDiversity,
    'Craig': Craig,
    'DeepFool': DeepFool,
    'Forgetting': Forgetting,
    'Full': Full,
    'Glister': Glister,
    'GraNd': GraNd,
    'Herding': Herding,
    'kCenterGreedy': kCenterGreedy,
    'Submodular': Submodular,
    'Uncertainty': Uncertainty,
    'Uniform': Uniform
}
