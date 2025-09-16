from rainbowneko.parser import CfgWDModelParser, neko_cfg
from rainbowneko.data import RatioBucket
from hcpdiff.data import TextImagePairDataset

@neko_cfg
def make_cfg():
    return dict(
        data_train=dict(
            dataset1=TextImagePairDataset(
                _partial_=True,
                batch_size=4,
                cache_latents=True,
                att_mask_encode=False,
                loss_weight=1.0,

                source=dict(
                    data_source1=None
                ),
                bucket=RatioBucket.from_files(
                    target_area=512*512,
                    num_bucket=5,
                ),
            )
        ),
    )