from io import BytesIO
from typing import Dict, Any, Tuple
import random
import lmdb
from PIL import Image

from rainbowneko.data.source.lmdb_img_label import LMDBImageLabelSource

class LmdbText2ImageSource(LMDBImageLabelSource):
    def __init__(self, img_root, label_file, prompt_template, repeat=1, **kwargs):
        super().__init__(img_root, label_file, repeat=repeat, **kwargs)
        self.prompt_template = self.load_template(prompt_template)
    
    def load_template(self, template_file):
        with open(template_file, 'r', encoding='utf-8') as f:
            return f.read().strip().split('\n')

    def _load_img_ids(self, label_dict):
        # no need to check whether x is img (remove `def is_image_file`)
        # return [x for x in label_dict.keys() if is_image_file(x)] * self.repeat
        return [x for x in label_dict.keys() if x] * self.repeat 

    def __getitem__(self, index) -> Dict[str, Any]:
        img_id = self.img_ids[index]
        with self.env.begin(write=False) as txn:
            img_data = txn.get(img_id.encode())  # get img data from LMDB
        image = Image.open(BytesIO(img_data))

        return {
            'id': img_id,
            'image': image,
            'prompt': {
                'template':random.choice(self.prompt_template),
                'caption':self.label_dict.get(img_id, None),
            }
        }

    def get_image_size(self, data: Dict[str, Any]) -> Tuple[int, int]:
        return data['image'].size