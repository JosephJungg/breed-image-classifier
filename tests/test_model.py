import torch

from src.model import build_model, count_params, set_train_mode, unfreeze_layer4


def test_probe_trains_only_the_head():
    model = build_model("resnet18", pretrained=False)
    assert count_params(model) == (512 * 37 + 37, 11_195_493)
    assert all(p.requires_grad for p in model.fc.parameters())
    assert not any(p.requires_grad for p in model.layer4.parameters())


def test_resnet50_head_has_2048_inputs():
    model = build_model("resnet50", pretrained=False)
    assert model.fc.in_features == 2048
    assert count_params(model)[0] == 2048 * 37 + 37


def test_unfreeze_layer4_leaves_earlier_layers_frozen():
    model = build_model("resnet18", pretrained=False)
    unfreeze_layer4(model)
    assert count_params(model)[0] == 8_412_709
    assert not any(p.requires_grad for p in model.layer3.parameters())


def test_set_train_mode_keeps_frozen_batchnorm_in_eval():
    model = build_model("resnet18", pretrained=False)
    set_train_mode(model)
    assert not model.bn1.training
    assert not model.layer3[0].bn1.training
    assert model.layer4[0].bn1.training
    assert model.fc.training


def test_output_has_37_scores():
    model = build_model("resnet18", pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 37)
