from dataclasses import dataclass, field
from typing import Optional

@dataclass
class StitchArguments:
    """
    Model stitching arguments
    """

    _argument_group_name = "Stitch Arguments"
    
    do_stitch: Optional[bool] = field(
        default=False, metadata={"help": "whether to stitch two source models"}
    )
    src_model1_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the first source pretrained model"},
    )
    src_model2_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the second source pretrained model"},
    )
    skip_layernorm: Optional[bool] = field(
        default=False, metadata={"help": "whether to skip layernorms"}
    )