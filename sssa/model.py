'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * References: clip
 * https://github.com/openai/CLIP
'''


import torch
import torch.nn as nn
import json
from tqdm import tqdm

import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List

import torch
from tqdm import tqdm

from clip.model import build_model
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from data.vocab import get_classifier_wordnet, get_classifier

_tokenizer = _Tokenizer()

_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}

import torch.nn.parallel as parallel

def dataparallel_custom(func):
    def wrapper(self, input_data, **kwargs):

        # Step 1: Scatter input_data across devices
        device_ids = self.devices
        replicas = parallel.replicate(self, device_ids)
        inputs_scattered = parallel.scatter(input_data, target_gpus=device_ids)
        # Step 2: Apply the function on each device
        replicas = replicas[:len(inputs_scattered)]
        outputs = [func(replica, input) for replica, input in zip(replicas, inputs_scattered)]

        # Step 3: Gather the results back to the main device
        output_data = parallel.gather(outputs, target_device=device_ids[0])
        
        return output_data
    return wrapper



def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None, mask: bool = False):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        print('===load nonjit===')
        model = build_model(state_dict or model.state_dict(), mask=mask).to(device)
        if str(device) == "cpu":
            model.float()
        return model

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class clip_classifier(nn.Module):
    
    def __init__(self, args, is_train=False):        
        super().__init__()

        self.templates = json.load(open(args.template,'r'))
        self.templates = self.templates[args.dataset]
        
        classnames = json.load(open(args.classname,'r'))
        self.classnames = classnames[args.dataset]
        
        self.model = load(args.clip_model,jit=False,mask=args.mask)
        self.model.float()
        self.init_classifier_weights(args) 
        
        self.model = self.model.to(args.device) 
        
        
    def init_classifier_weights(self, args):

        print(f"{len(self.classnames)} classes, {len(self.templates)} templates")

        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(self.classnames):
                texts = [template.format(classname) for template in self.templates] #format with class
                texts = tokenize(texts).to(args.device) #tokenize
                class_embeddings = self.model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
                
        self.model.visual.classifier = nn.Parameter(torch.stack(zeroshot_weights, dim=1).to(args.device))        
        
        # delete unused modules
        del self.model.transformer, self.model.token_embedding, self.model.positional_embedding, self.model.ln_final, self.model.text_projection, self.model.logit_scale 
        
        return

        
    def forward(self,images,**kwargs):
        return self.model.visual(images,**kwargs)
    
    
class clip_classifier_oszsl(nn.Module):
    def __init__(self, args, is_train=True):
        super().__init__()

        # self.templates = json.load(open(args.template,'r'))
        # self.templates = self.templates[args.dataset]
        
        # classnames = json.load(open(args.classname,'r'))
        # self.classnames = classnames[args.dataset]
        
        self.model = load(args.clip_model, device=args.device, jit=False, mask=args.mask)
        self.model.float()
        self.devices = [int(args.device.split(':')[-1])] if args.devices is None else args.devices
        if is_train:
            self.init_classifier_weights(args) 
            self.model = self.model.to(args.device)
            # if args.devices is not None:
            #     self.model = nn.DataParallel(self.model, args.devices)
            #     self.num_classes = self.model.module.visual.classifier.size(1)
            # else:
            self.num_classes = self.model.visual.classifier.size(1)
        return
        
        
    def init_classifier_weights(self, args):

        # print(f"{len(self.classnames)} classes, {len(self.templates)} templates")

        # with torch.no_grad():
        #     zeroshot_weights = []
        #     for classname in tqdm(self.classnames):
        #         texts = [template.format(classname) for template in self.templates] #format with class
        #         texts = tokenize(texts).to(args.device) #tokenize
        #         class_embeddings = self.model.encode_text(texts) #embed with text encoder
        #         class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        #         class_embedding = class_embeddings.mean(dim=0)
        #         class_embedding /= class_embedding.norm()
        #         zeroshot_weights.append(class_embedding)
        classifier = get_classifier(args)
        self.model.visual.classifier = nn.Parameter((classifier/classifier.norm(dim=-1, keepdim=True)).T, requires_grad=False)
        # self.model.visual.classifier = nn.utils.weight_norm(nn.Linear(in_features=classifier.size(0), out_features=classifier.size(1), requires_grad=args.classifier_requires_grad))
        # (classifier/classifier.norm(dim=-1, keepdim=True)).T
        # self.model.visual.classifier.weight_g.data.fill_(1)
        # self.model.visual.classifier.weight_g.requires_grad = False
        # delete unused modules
        # self.classifier_requires_grad = args.classifier_requires_grad
        
        del self.model.transformer, self.model.token_embedding, self.model.positional_embedding, self.model.ln_final, self.model.text_projection, self.model.logit_scale 
        
        return
    
    def get_visual_encoder(self):
        return self.model.module.visual if hasattr(self.model, 'module') else self.model.visual
    
    def forward(self, images, **kwargs):
        return self.get_visual_encoder()(images, **kwargs)
    
    def extract_vfeatures(self, images, **kwargs):
        return self.get_visual_encoder().extract_features(images,**kwargs)

