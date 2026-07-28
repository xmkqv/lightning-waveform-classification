from scripts import run
from scripts.config import LOG_DIR, REPORT_HTML, ModelConfig, TrainConfig
from waveform_classification import report
from waveform_classification.nets import bottleneck, dct, dppv, fcn, mlp, rproj, wavelet

MODEL_CONFIGS: list[ModelConfig] = [
    mlp.Config(),
    fcn.Config(),
    bottleneck.Config(),
    dct.Config(),
    dppv.Config(),
    rproj.Config(),
    wavelet.Config(),
]


def run_models(minimal: bool = False):
    seeds = [1] if minimal else [1, 2, 3, 4, 5]
    train = TrainConfig(max_epochs=2, patience=1) if minimal else TrainConfig()
    for model in MODEL_CONFIGS:
        cfg = run.Config(model=model, seeds=seeds, train=train)
        run.run_model(cfg)

    report.gen_report(LOG_DIR, report_path=REPORT_HTML)


if __name__ == "__main__":
    import sys

    minimal = "--minimal" in sys.argv
    run_models(minimal=minimal)
