from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import cv2
import PIL
import numpy as np 
from PIL import Image
import torch
from torchvision import transforms
from insightface.app import FaceAnalysis 
### insight-face installation can be found at https://github.com/deepinsight/insightface
from safetensors import safe_open
from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.utils import _get_model_file
from functions import process_text_with_markers, masks_for_unique_values, fetch_mask_raw_image, tokenize_and_mask_noun_phrases_ends, prepare_image_token_idx
from functions import ProjPlusModel, masks_for_unique_values
from attention import Consistent_IPAttProcessor, Consistent_AttProcessor, FacialEncoder
from data_util.face3d_helper import Face3DHelper
### Model can be imported from https://github.com/zllrunning/face-parsing.PyTorch?tab=readme-ov-file
### We use the ckpt of 79999_iter.pth: https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812
### Thanks for the open source of face-parsing model.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

# from modules.bn import InPlaceABNSync as BatchNorm2d

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
                )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer4(feat16) # 1/32
        return feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module,  nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


### This is not used, since I replace this with the resnet feature with the same size
class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        ## here self.sp is deleted
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)  # here return res3b1 feature
        feat_sp = feat_res8  # use res3b1 feature to replace spatial path feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, FeatureFusionModule) or isinstance(child, BiSeNetOutput):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


PipelineImageInput = Union[
    PIL.Image.Image,
    torch.FloatTensor,
    List[PIL.Image.Image],
    List[torch.FloatTensor],
]


class ConsistentIDStableDiffusionPipeline(StableDiffusionPipeline):
    
    @validate_hf_hub_args
    def load_ConsistentID_model(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: str,
        subfolder: str = '',
        trigger_word_ID: str = '<|image|>',
        trigger_word_facial: str = '<|facial|>',
        image_encoder_path: str = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',   # TODO Import CLIP pretrained model
        torch_dtype = torch.float16,
        num_tokens = 4,
        lora_rank= 128,
        **kwargs,
    ):
        self.lora_rank = lora_rank 
        self.torch_dtype = torch_dtype
        self.num_tokens = num_tokens
        self.set_ip_adapter()
        self.image_encoder_path = image_encoder_path
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=self.torch_dtype
        )   
        self.clip_image_processor = CLIPImageProcessor()
        self.id_image_processor = CLIPImageProcessor()
        self.crop_size = 512

        # FaceID
        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.similarity_threshold = 0.7  # Adjust this threshold as needed



        self.face3d_helper = Face3DHelper(bfm_dir='deep_3drecon/BFM', keypoint_mode='mediapipe', use_gpu=True)


        ### BiSeNet
        self.bise_net = BiSeNet(n_classes = 19)
        self.bise_net.cuda()
        self.bise_net_cp='./models/BiSeNet_pretrained_for_ConsistentID.pth' # Import BiSeNet model
        self.bise_net.load_state_dict(torch.load(self.bise_net_cp))
        self.bise_net.eval()
        # Colors for all 20 parts
        self.part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                    [255, 0, 85], [255, 0, 170],
                    [0, 255, 0], [85, 255, 0], [170, 255, 0],
                    [0, 255, 85], [0, 255, 170],
                    [0, 0, 255], [85, 0, 255], [170, 0, 255],
                    [0, 85, 255], [0, 170, 255],
                    [255, 255, 0], [255, 255, 85], [255, 255, 170],
                    [255, 0, 255], [255, 85, 255], [255, 170, 255],
                    [0, 255, 255], [85, 255, 255], [170, 255, 255]]
        
        ### LLVA Optional
        self.llva_model_path = "" #TODO import llava weights
        self.llva_prompt = "Describe this person's facial features for me, including face, ears, eyes, nose, and mouth." 
        self.llva_tokenizer, self.llva_model, self.llva_image_processor, self.llva_context_len = None,None,None,None #load_pretrained_model(self.llva_model_path)

        self.image_proj_model = ProjPlusModel(
            cross_attention_dim=self.unet.config.cross_attention_dim, 
            id_embeddings_dim=512,
            clip_embeddings_dim=self.image_encoder.config.hidden_size, 
            num_tokens=self.num_tokens,  # 4
        ).to(self.device, dtype=self.torch_dtype)
        self.FacialEncoder = FacialEncoder(self.image_encoder).to(self.device, dtype=self.torch_dtype)

        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"id_encoder": {}, "lora_weights": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("id_encoder."):
                            state_dict["id_encoder"][key.replace("id_encoder.", "")] = f.get_tensor(key)
                        elif key.startswith("lora_weights."):
                            state_dict["lora_weights"][key.replace("lora_weights.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict
    
        self.trigger_word_ID = trigger_word_ID
        self.trigger_word_facial = trigger_word_facial

        self.FacialEncoder.load_state_dict(state_dict["FacialEncoder"], strict=True)
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["adapter_modules"], strict=True)
        print(f"Successfully loaded weights from checkpoint")

        # Add trigger word token
        if self.tokenizer is not None: 
            self.tokenizer.add_tokens([self.trigger_word_ID], special_tokens=True)
            self.tokenizer.add_tokens([self.trigger_word_facial], special_tokens=True)

    def set_ip_adapter(self):
        unet = self.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = Consistent_AttProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=self.lora_rank,
                ).to(self.device, dtype=self.torch_dtype)
            else:
                attn_procs[name] = Consistent_IPAttProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0, rank=self.lora_rank, num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.torch_dtype)
        
        unet.set_attn_processor(attn_procs)



    import cv2
    import numpy as np

    @torch.inference_mode()
    def align_the_face(self, input_image, landmarks):
        # Define the target coordinates for the aligned face
        target_coords = np.array([
            [0.31556875, 0.4615741],  # Left eye
            [0.68262291, 0.4615741],  # Right eye
            [0.5, 0.6405965],         # Nose tip
            [0.34947187, 0.8246138],  # Left mouth corner
            [0.65073854, 0.8246138]   # Right mouth corner
        ], dtype=np.float32)

        # Convert landmarks to numpy array
        landmarks = np.array(landmarks, dtype=np.float32)

        # Select the relevant landmarks for alignment
        selected_landmarks = landmarks[[38, 88, 80, 84, 90]]

        # Compute the transformation matrix
        transformation_matrix, _ = cv2.estimateAffinePartial2D(selected_landmarks, target_coords, method=cv2.LMEDS)

        # Get the image dimensions
        height, width, _ = input_image.shape

        # Apply the transformation to the input image
        aligned_face = cv2.warpAffine(input_image, transformation_matrix, (width, height), borderValue=0.0)

        return aligned_face


    @torch.inference_mode()
    def detect_faces(self, input_image):
        faces = self.app.get(np.array(input_image))
        return faces

    @torch.inference_mode()
    def detect_facial_landmarks(self, input_image):
        faces = self.detect_faces(input_image)
        if len(faces) > 0:
            landmarks = faces[0].landmark_2d_106
            return landmarks
        else:
            return None

    @torch.inference_mode()
    def align_face(self, input_image):
        landmarks = self.detect_facial_landmarks(input_image)
        if landmarks is not None:
            # Perform face alignment using the detected landmarks
            aligned_face = self.align_the_face(np.array(input_image), landmarks)
            return aligned_face
        else:
            return input_image

    @torch.inference_mode()
    def extract_facial_features(self, input_image):
        faces = self.detect_faces(input_image)
        if len(faces) > 0:
            features = faces[0].normed_embedding
            return features
        else:
            return None

    @torch.inference_mode()
    def verify_face(self, input_image, generated_image):
        input_features = self.extract_facial_features(input_image)
        generated_features = self.extract_facial_features(generated_image)
        if input_features is not None and generated_features is not None:
            similarity = np.dot(input_features, generated_features)
            return similarity > self.similarity_threshold
        else:
            return False
    @torch.inference_mode()
    def get_facial_embeds(self, prompt_embeds, negative_prompt_embeds, facial_clip_images, facial_token_masks, valid_facial_token_idx_mask, lm3d):
        
        hidden_states = []
        uncond_hidden_states = []
        for facial_clip_image in facial_clip_images:
            hidden_state = self.image_encoder(facial_clip_image.to(self.device, dtype=self.torch_dtype), output_hidden_states=True).hidden_states[-2]
            uncond_hidden_state = self.image_encoder(torch.zeros_like(facial_clip_image, dtype=self.torch_dtype).to(self.device), output_hidden_states=True).hidden_states[-2]
            hidden_states.append(hidden_state)
            uncond_hidden_states.append(uncond_hidden_state)
        multi_facial_embeds = torch.stack(hidden_states)       
        uncond_multi_facial_embeds = torch.stack(uncond_hidden_states)   

        # Incorporate 3D facial landmarks
        lm3d_embeds = self.image_encoder(lm3d.to(self.device, dtype=self.torch_dtype), output_hidden_states=True).hidden_states[-2]
        multi_facial_embeds = torch.cat([multi_facial_embeds, lm3d_embeds.unsqueeze(0)], dim=1)
        uncond_multi_facial_embeds = torch.cat([uncond_multi_facial_embeds, lm3d_embeds.unsqueeze(0)], dim=1)

        # condition 
        facial_prompt_embeds = self.FacialEncoder(prompt_embeds, multi_facial_embeds, facial_token_masks, valid_facial_token_idx_mask)  

        # uncondition 
        uncond_facial_prompt_embeds = self.FacialEncoder(negative_prompt_embeds, uncond_multi_facial_embeds, facial_token_masks, valid_facial_token_idx_mask)  

        return facial_prompt_embeds, uncond_facial_prompt_embeds        

    @torch.inference_mode()   
    def get_image_embeds(self, faceid_embeds, face_image, s_scale, shortcut=False):

        clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.torch_dtype)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]
        
        faceid_embeds = faceid_embeds.to(self.device, dtype=self.torch_dtype)
        image_prompt_tokens = self.image_proj_model(faceid_embeds, clip_image_embeds, shortcut=shortcut, scale=s_scale)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(faceid_embeds), uncond_clip_image_embeds, shortcut=shortcut, scale=s_scale)
        
        return image_prompt_tokens, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, Consistent_IPAttProcessor):
                attn_processor.scale = scale

    @torch.inference_mode()
    def get_prepare_faceid(self, face_image):
        faceid_image = np.array(face_image)
        faces = self.app.get(faceid_image)
        if faces==[]:
            faceid_embeds = torch.zeros_like(torch.empty((1, 512)))
        else:
            faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

        return faceid_embeds

    @torch.inference_mode()
    def parsing_face_mask(self, raw_image_refer):

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        to_pil = transforms.ToPILImage()

        with torch.no_grad():
            # Ensure raw_image_refer is a PIL Image
            if isinstance(raw_image_refer, np.ndarray):
                image = Image.fromarray(raw_image_refer)
            else:
                image = raw_image_refer

            # Resize the image correctly
            image = image.resize((512, 512), resample=Image.BILINEAR) # Note: Image.BILINEAR not resample=Image.BILINEAR
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0).float().cuda()
            out = self.bise_net(img)[0]
            parsing_anno = out.squeeze(0).cpu().numpy().argmax(0)

        im = np.array(image)
        vis_im = im.copy().astype(np.uint8)
        stride = 1
        vis_parsing_anno = cv2.resize(parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)
        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = self.part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        vis_parsing_anno_color = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

        return vis_parsing_anno_color, vis_parsing_anno


    @torch.inference_mode()
    def get_prepare_llva_caption(self, input_image_file, model_path=None, prompt=None):
        
        ### Optional: Use the LLaVA
        # args = type('Args', (), {
        #     "model_path": self.llva_model_path,
        #     "model_base": None,
        #     "model_name": get_model_name_from_path(self.llva_model_path),
        #     "query": self.llva_prompt,
        #     "conv_mode": None,
        #     "image_file": input_image_file,
        #     "sep": ",",
        #     "temperature": 0,
        #     "top_p": None,
        #     "num_beams": 1,
        #     "max_new_tokens": 512
        # })() 
        # face_caption = eval_model(args, self.llva_tokenizer, self.llva_model, self.llva_image_processor)

        ### Use built-in template
        face_caption = "The person has one nose, two eyes, two ears, and a mouth."

        return face_caption

    @torch.inference_mode()
    def get_prepare_facemask(self, input_image_file):

        aligned_face = self.align_face(input_image_file)
        vis_parsing_anno_color, vis_parsing_anno = self.parsing_face_mask(aligned_face)
   
        parsing_mask_list = masks_for_unique_values(vis_parsing_anno) 

        key_parsing_mask_list = {}
        key_list = ["Face", "Left_Ear", "Right_Ear", "Left_Eye", "Right_Eye", "Nose", "Upper_Lip", "Lower_Lip"]
        processed_keys = set()
        for key, mask_image in parsing_mask_list.items():
            if key in key_list:
                if "_" in key:
                    prefix = key.split("_")[1]
                    if prefix in processed_keys:                   
                        continue
                    else:            
                        key_parsing_mask_list[key] = mask_image 
                        processed_keys.add(prefix)  
            
                key_parsing_mask_list[key] = mask_image            

        # Extract 3D facial landmarks
        coeff_dict = self.face3d_helper.estimate_coeff(input_image_file)
        lm3d = self.face3d_helper.reconstruct_lm3d(
            torch.tensor(coeff_dict['id']).cuda(),
            torch.tensor(coeff_dict['exp']).cuda(),
            torch.tensor(coeff_dict['euler']).cuda(),
            torch.tensor(coeff_dict['trans']).cuda()
        )
        return key_parsing_mask_list, vis_parsing_anno_color, lm3d



    def encode_prompt_with_trigger_word(
        self,
        prompt: str,
        face_caption: str,
        key_parsing_mask_list = None,
        image_token = "<|image|>", 
        facial_token = "<|facial|>",
        max_num_facials = 5,
        num_id_images: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        face_caption_align, key_parsing_mask_list_align = process_text_with_markers(face_caption, key_parsing_mask_list) 
        
        prompt_face = prompt + "Detail:" + face_caption_align

        max_text_length=330      
        if len(self.tokenizer(prompt_face, max_length=self.tokenizer.model_max_length, padding="max_length",truncation=False,return_tensors="pt").input_ids[0])!=77:
            prompt_face = "Detail:" + face_caption_align + " Caption:" + prompt
        
        if len(face_caption)>max_text_length:
            prompt_face = prompt
            face_caption_align =  ""
  
        prompt_text_only = prompt_face.replace("<|facial|>", "").replace("<|image|>", "")
        tokenizer = self.tokenizer
        facial_token_id = tokenizer.convert_tokens_to_ids(facial_token)
        image_token_id = None

        clean_input_id, image_token_mask, facial_token_mask = tokenize_and_mask_noun_phrases_ends(
        prompt_face, image_token_id, facial_token_id, tokenizer) 

        image_token_idx, image_token_idx_mask, facial_token_idx, facial_token_idx_mask = prepare_image_token_idx(
            image_token_mask, facial_token_mask, num_id_images, max_num_facials )

        return prompt_text_only, clean_input_id, key_parsing_mask_list_align, facial_token_mask, facial_token_idx, facial_token_idx_mask

    @torch.inference_mode()
    def get_prepare_clip_image(self, input_image_file, key_parsing_mask_list, image_size=512, max_num_facials=5, change_facial=True):
        
        facial_mask = []
        facial_clip_image = []
        transform_mask = transforms.Compose([transforms.CenterCrop(size=image_size), transforms.ToTensor(),])
        clip_image_processor = CLIPImageProcessor()

        num_facial_part = len(key_parsing_mask_list)

        for key in key_parsing_mask_list:
            key_mask=key_parsing_mask_list[key]
            facial_mask.append(transform_mask(key_mask))
            key_mask_raw_image = fetch_mask_raw_image(input_image_file,key_mask)
            parsing_clip_image = clip_image_processor(images=key_mask_raw_image, return_tensors="pt").pixel_values
            facial_clip_image.append(parsing_clip_image)

        padding_ficial_clip_image = torch.zeros_like(torch.zeros([1, 3, 224, 224]))
        padding_ficial_mask = torch.zeros_like(torch.zeros([1, image_size, image_size]))

        if num_facial_part < max_num_facials:
            facial_clip_image += [torch.zeros_like(padding_ficial_clip_image) for _ in range(max_num_facials - num_facial_part) ]
            facial_mask += [ torch.zeros_like(padding_ficial_mask) for _ in range(max_num_facials - num_facial_part)]

        facial_clip_image = torch.stack(facial_clip_image, dim=1).squeeze(0)
        facial_mask = torch.stack(facial_mask, dim=0).squeeze(dim=1)

        return facial_clip_image, facial_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        target_size: Optional[Tuple[int, int]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        input_id_images: PipelineImageInput = None,
        start_merge_step: int = 0,
        class_tokens_mask: Optional[torch.LongTensor] = None,
        prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )
        if not isinstance(input_id_images, list):
            input_id_images = [input_id_images]

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale >= 1.0
        input_image_file = input_id_images[0]

        faceid_embeds = self.get_prepare_faceid(face_image=input_image_file)
        face_caption = self.get_prepare_llva_caption(input_image_file)
        key_parsing_mask_list, vis_parsing_anno_color, lm3d = self.get_prepare_facemask(input_image_file)
   
        # faceid_embeds = self.get_prepare_faceid(face_image=input_image_file)
        # face_caption = self.get_prepare_llva_caption(input_image_file)
        # key_parsing_mask_list, vis_parsing_anno_color = self.get_prepare_facemask(input_image_file)

        assert do_classifier_free_guidance

        # 3. Encode input prompt
        num_id_images = len(input_id_images)

        (
            prompt_text_only,
            clean_input_id,
            key_parsing_mask_list_align,
            facial_token_mask,
            facial_token_idx,
            facial_token_idx_mask,
        ) = self.encode_prompt_with_trigger_word(
            prompt = prompt,
            face_caption = face_caption,
            # prompt_2=None,  
            key_parsing_mask_list=key_parsing_mask_list,
            device=device,
            max_num_facials = 5,
            num_id_images= num_id_images,
            # prompt_embeds= None,
            # pooled_prompt_embeds= None,
            # class_tokens_mask= None,
        )

        # 4. Encode input prompt without the trigger word for delayed conditioning
        encoder_hidden_states = self.text_encoder(clean_input_id.to(device))[0] 

       
        prompt_embeds = self._encode_prompt(
            prompt_text_only,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        negative_encoder_hidden_states_text_only = prompt_embeds[0:num_images_per_prompt]
        encoder_hidden_states_text_only = prompt_embeds[num_images_per_prompt:]

        # 5. Prepare the input ID images
        prompt_tokens_faceid, uncond_prompt_tokens_faceid = self.get_image_embeds(faceid_embeds, face_image=input_image_file, s_scale=1.0, shortcut=False)

        facial_clip_image, facial_mask = self.get_prepare_clip_image(input_image_file, key_parsing_mask_list_align, image_size=512, max_num_facials=5)
        facial_clip_images = facial_clip_image.unsqueeze(0).to(device, dtype=self.torch_dtype)
        facial_token_mask = facial_token_mask.to(device)
        facial_token_idx_mask = facial_token_idx_mask.to(device)
        negative_encoder_hidden_states = negative_encoder_hidden_states_text_only

        cross_attention_kwargs = {}

        # 6. Get the update text embedding
        # prompt_embeds_facial, uncond_prompt_embeds_facial = self.get_facial_embeds(encoder_hidden_states, negative_encoder_hidden_states, \
                                                            # facial_clip_images, facial_token_mask, facial_token_idx_mask)
        prompt_embeds_facial, uncond_prompt_embeds_facial = self.get_facial_embeds(encoder_hidden_states, negative_encoder_hidden_states, facial_clip_images, facial_token_mask, facial_token_idx_mask, lm3d)


        prompt_embeds = torch.cat([prompt_embeds_facial, prompt_tokens_faceid], dim=1)
        negative_prompt_embeds = torch.cat([uncond_prompt_embeds_facial, uncond_prompt_tokens_faceid], dim=1)

        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )        
        prompt_embeds_text_only = torch.cat([encoder_hidden_states_text_only, prompt_tokens_faceid], dim=1)
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_text_only], dim=0)

        # 7. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 8. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        (
            null_prompt_embeds,
            augmented_prompt_embeds,
            text_prompt_embeds,
        ) = prompt_embeds.chunk(3)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                if i <= start_merge_step:
                    current_prompt_embeds = torch.cat(
                        [null_prompt_embeds, text_prompt_embeds], dim=0
                    )
                else:
                    current_prompt_embeds = torch.cat(
                        [null_prompt_embeds, augmented_prompt_embeds], dim=0
                    )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                else:
                    assert 0, "Not Implemented"

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 9.1 Post-processing
            image = self.decode_latents(latents)

            # 9.2 Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )

            # 9.3 Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 9.1 Post-processing
            image = self.decode_latents(latents)

            # 9.2 Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        # 9.4 Verify the generated face
        generated_image = self.decode_latents(latents)
        if not self.verify_face(input_image_file, generated_image):
            print("Warning: Generated face does not match the input identity.")

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )









