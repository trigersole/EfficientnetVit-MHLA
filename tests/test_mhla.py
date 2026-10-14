import unittest

import torch

from efficientvit.models.nn import LiteMLA, MultiHeadLinearAttention
from efficientvit.seg_model_zoo import create_efficientvit_seg_experiment_model


class MultiHeadLinearAttentionTest(unittest.TestCase):
    def test_m1_matches_standard_normalized_linear_attention(self):
        torch.manual_seed(0)
        q = torch.relu(torch.randn(2, 3, 4, 5, 7))
        k = torch.relu(torch.randn(2, 3, 4, 5, 7))
        v = torch.randn(2, 3, 4, 5, 7)
        mhla = MultiHeadLinearAttention(num_groups=1, eps=1.0e-6)
        actual = mhla(q, k, v)

        q_tokens = q.flatten(3).transpose(-1, -2)
        k_tokens = k.flatten(3).transpose(-1, -2)
        v_tokens = v.flatten(3).transpose(-1, -2)
        summary = torch.einsum("bhnd,bhne->bhde", k_tokens, v_tokens)
        normalizer = k_tokens.sum(dim=2)
        expected = torch.einsum("bhnd,bhde->bhne", q_tokens, summary)
        expected = expected / (
            torch.einsum("bhnd,bhd->bhn", q_tokens, normalizer).unsqueeze(-1) + 1.0e-6
        )
        expected = expected.transpose(-1, -2).reshape_as(actual)
        torch.testing.assert_close(actual, expected)

    def test_arbitrary_spatial_shape_and_gradients(self):
        for num_groups in (2, 4, 8):
            with self.subTest(num_groups=num_groups):
                torch.manual_seed(1)
                module = MultiHeadLinearAttention(num_groups=num_groups, eps=1.0e-6)
                q = torch.relu(torch.randn(3, 2, 8, 5, 7, requires_grad=True))
                k = torch.relu(torch.randn(3, 2, 8, 5, 7, requires_grad=True))
                v = torch.randn(3, 2, 8, 5, 7, requires_grad=True)
                out = module(q, k, v)
                self.assertEqual(out.shape, v.shape)
                self.assertTrue(torch.isfinite(out).all())
                out.square().mean().backward()
                self.assertIsNotNone(module.mixing_weight.grad)
                self.assertTrue(torch.isfinite(module.mixing_weight.grad).all())


class LiteMLAVariantTest(unittest.TestCase):
    def test_e0_to_e3_shapes_backward_and_no_nan(self):
        variants = {
            "e0": ("lite_mla", True),
            "e1": ("lite_mla", False),
            "e2": ("mhla", False),
            "e3": ("mhla", True),
        }
        for name, (attention_type, use_multiscale_qkv) in variants.items():
            with self.subTest(name=name):
                block = LiteMLA(
                    in_channels=32,
                    out_channels=32,
                    heads=2,
                    dim=8,
                    attention_type=attention_type,
                    use_multiscale_qkv=use_multiscale_qkv,
                    mhla_num_groups=4,
                )
                x = torch.randn(2, 32, 7, 11, requires_grad=True)
                y = block(x)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all())
                y.mean().backward()
                self.assertIsNotNone(block.qkv.conv.weight.grad)
                if attention_type == "mhla":
                    self.assertIsNotNone(block.mhla.mixing_weight.grad)
                self.assertEqual(len(block.aggreg), 1 if use_multiscale_qkv else 0)

    def test_full_segmentation_variants(self):
        for experiment in ("e0", "e1", "e2", "e3"):
            with self.subTest(experiment=experiment):
                model = create_efficientvit_seg_experiment_model(
                    name="efficientvit-seg-b0-cityscapes",
                    experiment=experiment,
                    pretrained=False,
                ).eval()
                with torch.inference_mode():
                    output = model(torch.randn(1, 3, 64, 96))
                self.assertEqual(output.shape, (1, 19, 8, 12))
                self.assertTrue(torch.isfinite(output).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for fp16 AMP")
    def test_cuda_amp(self):
        block = LiteMLA(
            in_channels=32,
            out_channels=32,
            heads=2,
            dim=8,
            attention_type="mhla",
            use_multiscale_qkv=False,
        ).cuda()
        x = torch.randn(2, 32, 8, 8, device="cuda", requires_grad=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = block(x)
            loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(block.mhla.mixing_weight.grad)


if __name__ == "__main__":
    unittest.main()
