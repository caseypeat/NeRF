
for i in {1..10}:
do
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C1_0.yaml
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C1_1.yaml
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C1_2.yaml
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C1_3.yaml
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C1_4.yaml
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C1_5.yaml
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C1_6.yaml
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C1_7.yaml

    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C2_0.yaml
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C2_1.yaml
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C2_2.yaml
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C2_3.yaml
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C2_4.yaml
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C2_5.yaml
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C2_6.yaml
    python -m allign.measure.allign_random_error --configs ./allign/configs/config_random.yaml ./allign/configs/full/vine_C2_7.yaml
done