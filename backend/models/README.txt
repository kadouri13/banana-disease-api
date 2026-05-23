Place the following model files in this directory before starting the server:

  model_custom.h5    — Custom CNN (trained on Kaggle)
  model_resnet.h5    — ResNet50 (trained on Kaggle)
  model_inception.h5 — InceptionV3 (trained on Kaggle)

To export them from your Kaggle notebook, run the export_models.py helper
or add this cell at the end of your notebook:

    import os
    os.makedirs("/kaggle/working/models", exist_ok=True)
    model_custom.save("/kaggle/working/models/model_custom.h5")
    model_res.save("/kaggle/working/models/model_resnet.h5")
    model_inc.save("/kaggle/working/models/model_inception.h5")
    print("Done — download from Kaggle Output tab")
