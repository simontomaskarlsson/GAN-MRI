### When generating synthetic images from a trained models:

#### Put models in directory:
```
./models/
```

#### With names:
```
G_A2B_model.hdf5
G_B2A_model.hdf5
```

#### Create directories for generated images:
```
./synthetic_images/A
./synthetic_images/B
```

#### Comment row 242:
```
#self.train(…
```

#### Uncomment row 243:
```
self.load_model_and_generate_synthetic_images()
```

#### Then run:
```
python CycleGAN.py
```
