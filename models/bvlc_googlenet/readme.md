---
name: BVLC GoogleNet Model
license: unrestricted
---

This model is a replication of the model described in the [GoogleNet](http://arxiv.org/abs/1409.4842) publication. We would like to thank Christian Szegedy for all his help in the replication of GoogleNet model.

Differences:
- not training with the relighting data-augmentation;
- not training with the scale or aspect-ratio data-augmentation;
- uses "xavier" to initialize the weights instead of "gaussian";
- quick_solver.prototxt uses a different learning rate decay policy than the original solver.prototxt, that allows a much faster training (60 epochs vs 250 epochs);

This model obtains a top-1 accuracy 68.7% (31.3% error) and a top-5 accuracy 88.9% (11.1% error) on the validation set, using just the center crop.

This model was trained by Sergio Guadarrama @sguada

## License

This model is released for unrestricted use.
