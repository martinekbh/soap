# SOaP for positional noise in ViT embeddings

This repo contains code for the paper **MIM Representations Encode Non-Semantic Noise: Post-Hoc Suppression Boosts Zero-Shot Performance**.

## Loading models and SOaP

SOaP can be attached to any model as a single linear projection layer.
Either follow the instructions in `notebook.ipynb` to fit the projection to a pretrained model of your choice, or download checkpoints from XXX.
NB: Checkpoints will be released with the released non-anonymous repo.
The weights should be placed into a `weights` folder in the cloned repo.

Example for CAPI
```
from get_models import get_dense_backbone
from soap.soap import SemanticInvarianceProjector

modelname = 'capi_large'
features = 'out'
model = get_dense_backbone(modelname).eval()
projector = SemanticInvarianceProjector.from_modelname(modelname, 'weights')
```

## More examples
We provide a [jupyter notebook](notebook.ipynb) that illustrates loading, fitting the projector, and evaluating on salient segmentation.