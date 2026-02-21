import yaml


def main() -> None:
    with open("configs/config.yaml", "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    data_cfg = cfg.get("data", {})
    label_cfg = cfg.get("label", {})
    print(
        "Config loaded successfully | "
        f"symbol={data_cfg.get('symbol')} "
        f"timeframe={data_cfg.get('timeframe')} "
        f"horizon_k={label_cfg.get('horizon_k')} "
        f"strongmove_atr_mult={label_cfg.get('strongmove_atr_mult')}"
    )


if __name__ == "__main__":
    main()
