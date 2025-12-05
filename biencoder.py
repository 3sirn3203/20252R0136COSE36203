import argparse
from src.trainer.biencoder_trainer import train_biencoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bi-encoder with triplet loss.")
    parser.add_argument("--config", type=str, default="src/config/bi_encoder_config.json", help="Path to config JSON.")
    parser.add_argument("--local-test", action="store_true", help="Use small subset for quick testing.")
    args = parser.parse_args()

    train_biencoder(args.config, local_test=args.local_test)
