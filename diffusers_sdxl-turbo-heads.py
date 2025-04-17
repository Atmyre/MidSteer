import warnings
warnings.filterwarnings('ignore')

from typing import Optional, Union, Tuple, List, Callable, Any, Dict
from einops import rearrange, repeat
import abc
import numpy as np
import pickle
from PIL import Image
from collections import defaultdict
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import torch.nn.functional as F

from tqdm import tqdm

device = 'cuda'

pipe = StableDiffusionXLPipeline.from_pretrained(
    "../camera2/atmyre/cache/models--stabilityai--sdxl-turbo/snapshots/71153311d3dbb46851df1931d3ca6e939de83304/", 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16",
    safety_checker=None
    )
pipe.to("cuda")

# Define Controller for BasicTransformerBlock
class AttentionControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, vector, num_heads, idx):
        
        vector = self.forward(vector, num_heads, idx)
        
#         self.cur_att_layer += 1
#         if self.cur_att_layer == self.num_att_layers:
#             self.cur_att_layer = 0
#             self.cur_step += 1
#             self.between_steps()
        return vector


class EmptyControl(AttentionControl):
    def forward (self, vector, place_in_unet: str):
        return vector


class AttentionStore(AttentionControl):
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.attn_layer_num = 0
        self.step_store = self.get_empty_store()
        self.vector_store = defaultdict(dict)
        self.steer=True
        self.num_steer = 1
        self.steer_back = False
        self.intensity=30

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.vector_store = defaultdict(dict)
        
    @staticmethod
    def get_empty_store():
        return defaultdict(list)

    def forward(self, vector, num_heads, idx=0):
            
        # save activation (vector) for further computing steering vectors
#         print(vector.data.cpu().numpy().shape)
        self.step_store[self.attn_layer_num].append(vector.data.cpu().numpy()[len(vector)//2:].mean(axis=0).mean(axis=0))
        
#         37, 18 color
#         37, 19 spiral
#         if self.attn_layer_num in [36] and idx==0:
#             print(np.linalg.norm(vector.data.cpu().numpy()))
# #             emb_size = vector.shape[2]//num_heads
#             vector = 20*vector
#         else:
#             vector = torch.tensor(np.zeros_like(vector.data.cpu().numpy())).to(device)
            
        if (self.attn_layer_num==28 and idx==17):
            vector = 100*vector
        else:
            vector = 1*vector

#         if self.attn_layer_num == 11:
#             vector = 100*vector
    
        if idx == num_heads-1:
            self.attn_layer_num += 1
        
        return vector

    def between_steps(self):
        self.vector_store[self.cur_step] = self.step_store
        self.step_store = self.get_empty_store()
        self.step_store_size = 0
    
from copy import deepcopy

class CustomAttnProcessor:
     
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
            
        print(encoder_hidden_states)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

         # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)


        # -------------------------------
        # adding controller
        hidden_states = controller(hidden_states, attn.heads)
        
        
#         size = hidden_states.shape[2] // attn.heads
# #         y = torch.tensor(np.zeros_like(hidden_states.data.cpu().numpy())).to(device)
#         for idx in range(attn.heads):
# #             x = deepcopy(y[:, :, :])
# #             x[:, :, size*idx:size*(idx+1)] = hidden_states[:, :, size*idx:size*(idx+1)]
            
#             x = hidden_states[:, :, size*idx:size*(idx+1)]
            
# #             x = attn.to_out[0](x)
# #             # dropout
# #             x = attn.to_out[1](x)
            
# #             if attn.residual_connection:
# #                 x = x + residual

# #             x = x / attn.rescale_output_factor
            
#             x = controller(x, attn.heads, idx)
#             hidden_states[:, :, size*idx:size*(idx+1)] = x
        # -------------------------------
            

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    
# down 0 10                                                                                                          [82/1857]
# down 1 10
# down 2 10
# down 3 10
# down 4 20
# down 5 20
# down 6 20
# down 7 20
# down 8 20
# down 9 20
# down 10 20
# down 11 20
# down 12 20
# down 13 20
# down 14 20
# down 15 20
# down 16 20
# down 17 20
# down 18 20
# down 19 20
# down 20 20
# down 21 20
# down 22 20
# down 23 20
# up 0 20
# up 1 20
# up 2 20
# up 3 20
# up 4 20
# up 5 20
# up 6 20
# up 7 20
# up 8 20
# up 9 20
# up 10 20
# up 11 20
# up 12 20
# up 13 20
# up 14 20
# up 15 20
# up 16 20
# up 17 20
# up 18 20
# up 19 20
# up 20 20
# up 21 20
# up 22 20
# up 23 20
# up 24 20
# up 25 20
# up 26 20
# up 27 20
# up 28 20
# up 29 20
# up 30 10
# up 31 10
# up 32 10
# up 33 10
# up 34 10
# up 35 10

    
    
def register_vector_control(model, controller):
    
    def attn_forward(self, place_in_unet):
        # overriding BasicTransformerBlock forward function
       

        return __call__

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'BasicTransformerBlock':
            #print("register_recr", net_.__class__.__name__)
#             net_.attn2.processor.__call__ = attn_forward(net_, place_in_unet)
            net_.attn2.set_processor(CustomAttnProcessor())
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    attn_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            attn_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            attn_count += register_recr(net[1], 0, "up")
    controller.num_att_layers = attn_count
    

# vocab_tokens_clean = []
# with open ('vocab_clean.txt', 'r') as file:
#     for x in file.readlines():
#         vocab_tokens_clean.append(x.strip())
# file.close()
# print('vocab len:', len(vocab_tokens_clean))


# imagenet_classes = []
# imagenet_classes_f = open('../camera2/atmyre/generative-models/data/imagenet_classes.txt', 'r')
# for line in imagenet_classes_f.readlines():
#     imagenet_classes.append(line.strip().split(', ')[1])
    
# imagenet_templates = [
#       "a photo of a {}",
#       "a rendering of a {}",
#       "a cropped photo of the {}",
#       "the photo of a {}",
#       "a photo of a clean {}",
#       "a photo of a dirty {}",
#       "a dark photo of the {}",
#       "a photo of my {}",
#       "a photo of the cool {}",
#       "a close-up photo of a {}",
#       "a bright photo of the {}",
#       "a cropped photo of a {}",
#       "a photo of the {}",
#       "a good photo of the {}",
#       "a photo of one {}",
#       "a close-up photo of the {}",
#       "a rendition of the {}",
#       "a photo of the clean {}",
#       "a rendition of a {}",
#       "a photo of a nice {}",
#       "a good photo of a {}",
#       "a photo of the nice {}",
#       "a photo of the small {}",
#       "a photo of the weird {}",
#       "a photo of the large {}",
#       "a photo of a cool {}",
#       "a photo of a small {}",
#   ]
    
# print('generating neg vectors')
# neg_vectors = []
# for i in range(20):
    
#     controller = AttentionStore()
#     controller.steer=False
#     register_vector_control(pipe.unet, controller)

#     g = torch.Generator(device="cuda")
#     seed=i
#     g.manual_seed(seed)

#     image = pipe(prompt=imagenet_templates[i].format(''), 
#                  num_inference_steps=1, 
#                  guidance_scale=0.0,
#                  generator=g,
#                 ).images[0]

#     neg_vectors.append(controller.step_store)

# tokens = []
# with open('./data_xl_turbo/from_clip_words.txt', 'r') as f:
#     for line in f.readlines():
#         tokens.append(line.strip())
        
# f.close()
    
# # num_tokens = len(vocab_tokens_clean)//4
# num_tokens = len(tokens) // 8
# print(len(tokens), num_tokens)

# for token_num, token in enumerate(tokens[num_tokens*0:num_tokens*1]):
#     print(token_num, token)
# # for concept_num, concept in enumerate(['Snoopy', 'dog', 'rose']):

#     pos_vectors = []
#     neg_vectors = []
# #     pos_samples = []
# #     neg_samples = []
# #     print(token_num, vocab_token)

#     for i, cls in enumerate(imagenet_classes[:50]):
# #     for i in range(20):
#         for c in ['']:

#             controller = AttentionStore()
#             controller.steer=False
#             register_vector_control(pipe.unet, controller)

#             g = torch.Generator(device="cuda")
#             seed=i
#             g.manual_seed(seed)

#             image = pipe(prompt=cls+' with {}'.format(token), 
#                          num_inference_steps=1, 
#                          guidance_scale=0.0,
#                          generator=g,
#                         ).images[0]

#             pos_vectors.append(controller.step_store)
# #             pos_samples.append(image)

#             controller = AttentionStore()
#             controller.steer=False
#             register_vector_control(pipe.unet, controller)

#             g = torch.Generator(device="cuda")
#             seed=i
#             g.manual_seed(seed)

#             image = pipe(prompt=cls, 
#                          num_inference_steps=1, 
#                          guidance_scale=0.0,
#                          generator=g,
#                         ).images[0]

#             neg_vectors.append(controller.step_store)
# #             neg_samples.append(image)

# #     for iter_num in range(1, 2):
#     steering_vectors_gen = defaultdict(list)

#     for layer_num in range(len(pos_vectors[0])):

#         for attn_head_num in range(len(pos_vectors[0][layer_num])):

#             pos_0 = [pos_vectors[i][layer_num][attn_head_num] for i in range(len(pos_vectors))]
#             pos_avg_0 = np.mean(pos_0, axis=0)

#             neg_0 = [neg_vectors[i][layer_num][attn_head_num] for i in range(len(neg_vectors))]
#             neg_avg_0 = np.mean(neg_0, axis=0)


#             x = pos_avg_0 - neg_avg_0
# #                 x = x / np.linalg.norm(x)

#             steering_vectors_gen[layer_num].append(x)

# #     print(steering_vectors_gen[iter_num][key][0].shape)


#     # Saving steering vectors:
#     with open('./data_xl_turbo/words_vectors_heads_unnormed/{}_{}.pickle'.format("sdxl", token), 'wb') as handle:
#         pickle.dump(steering_vectors_gen, handle)


# desired_concept = 'president'
# with open('./data_xl/{}_{}.pickle'.format("sdxl", desired_concept), 'rb') as handle:
#     steering_vectors_gen = pickle.load(handle)

# Using steering vector:
prompt = "Leonaro DiCaprio drinking coffee"
controller = AttentionStore()
controller.steer=False
g = torch.Generator(device="cuda")
g.manual_seed(2)
register_vector_control(pipe.unet, controller)
image = pipe("", num_inference_steps=1, guidance_scale=0.0, generator=g).images[0]
# image.save("results-sdxl-turbo/test/orig_{}.jpg".format(prompt))
image.save("results-sdxl-turbo/test/heads_{}.jpg".format(prompt))


# prompt = "a girl carrying a bread loaf"
# controller = AttentionStore()
# controller.steer=True
# # steering backward, i.e. removing concept
# controller.steer_back=True
# controller.intensity=2
# register_vector_control(pipe.unet, controller)
# image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
# image.save("data_xl_turbo/imgs/forward_{}.jpg".format(prompt))

# imagenet_templates = [
#     'a bad photo of a {}.',
#     'a photo of many {}.',
#     'a sculpture of a {}.',
#     'a photo of the hard to see {}.',
#     'a low resolution photo of the {}.',
#     'a rendering of a {}.',
#     'graffiti of a {}.',
#     'a bad photo of the {}.',
#     'a cropped photo of the {}.',
#     'a tattoo of a {}.',
#     'the embroidered {}.',
#     'a photo of a hard to see {}.',
#     'a bright photo of a {}.',
#     'a photo of a clean {}.',
#     'a photo of a dirty {}.',
#     'a dark photo of the {}.',
#     'a drawing of a {}.',
#     'a photo of my {}.',
#     'the plastic {}.',
#     'a photo of the cool {}.',
#     'a close-up photo of a {}.',
#     'a black and white photo of the {}.',
#     'a painting of the {}.',
#     'a painting of a {}.',
#     'a pixelated photo of the {}.',
#     'a sculpture of the {}.',
#     'a bright photo of the {}.',
#     'a cropped photo of a {}.',
#     'a plastic {}.',
#     'a photo of the dirty {}.',
#     'a jpeg corrupted photo of a {}.',
#     'a blurry photo of the {}.',
#     'a photo of the {}.',
#     'a good photo of the {}.',
#     'a rendering of the {}.',
#     'a {} in a video game.',
#     'a photo of one {}.',
#     'a doodle of a {}.',
#     'a close-up photo of the {}.',
#     'a photo of a {}.',
#     'the origami {}.',
#     'the {} in a video game.',
#     'a sketch of a {}.',
#     'a doodle of the {}.',
#     'a origami {}.',
#     'a low resolution photo of a {}.',
#     'the toy {}.',
#     'a rendition of the {}.',
#     'a photo of the clean {}.',
#     'a photo of a large {}.',
#     'a rendition of a {}.',
#     'a photo of a nice {}.',
#     'a photo of a weird {}.',
#     'a blurry photo of a {}.',
#     'a cartoon {}.',
#     'art of a {}.',
#     'a sketch of the {}.',
#     'a embroidered {}.',
#     'a pixelated photo of a {}.',
#     'itap of the {}.',
#     'a jpeg corrupted photo of the {}.',
#     'a good photo of a {}.',
#     'a plushie {}.',
#     'a photo of the nice {}.',
#     'a photo of the small {}.',
#     'a photo of the weird {}.',
#     'the cartoon {}.',
#     'art of the {}.',
#     'a drawing of the {}.',
#     'a photo of the large {}.',
#     'a black and white photo of a {}.',
#     'the plushie {}.',
#     'a dark photo of a {}.',
#     'itap of a {}.',
#     'graffiti of the {}.',
#     'a toy {}.',
#     'itap of my {}.',
#     'a photo of a cool {}.',
#     'a photo of a small {}.',
#     'a tattoo of the {}.',
# ]


# for concept in ['snoopy', 'mickey', 'spongebob', 'pikachu', 'dog', 'legislator']:
#     for tmp in imagenet_templates:
#         for i in range(10):
#             prompt = tmp.format(concept)

#             controller = AttentionStore()
#             controller.steer=False
#             register_vector_control(pipe.unet, controller)

#             g = torch.Generator(device="cuda")
#             g.manual_seed(i)

#             images = pipe(prompt,
#                          num_images_per_prompt=1,
#                          num_inference_steps=30, 
#                          generator=g).images

#             images[0].save(
#                 './results-sdxl/orig/{}/{}.png'.format(concept, tmp.format(concept)[:-1]+'_'+str(i))
#                 )

#             controller = AttentionStore()
#             controller.steer=True
#             controller.steer_back=True
#             controller.intensity=4.
#             register_vector_control(pipe.unet, controller)

#             g = torch.Generator(device="cuda")
#             g.manual_seed(i)

#             images = pipe(prompt,
#                          num_images_per_prompt=1,
#                          num_inference_steps=30, 
#                           generator=g).images

#             images[0].save(
#                 './results-sdxl/rem_snoopy-4/{}/{}.png'.format(concept, tmp.format(concept)[:-1]+'_'+str(i))
#                 )

#             controller = AttentionStore()
#             controller.steer=True
#             controller.steer_back=True
#             controller.intensity=2.
#             register_vector_control(pipe.unet, controller)

#             g = torch.Generator(device="cuda")
#             g.manual_seed(i)

#             images = pipe(prompt,
#                          num_images_per_prompt=1,
#                          num_inference_steps=30, 
#                           generator=g).images

#             images[0].save(
#                 './results-sdxl/rem_snoopy-2/{}/{}.png'.format(concept, tmp.format(concept)[:-1]+'_'+str(i))
#                 )


#             controller = AttentionStore()
#             controller.steer=True
#             controller.steer_back=True
#             controller.intensity=3.
#             register_vector_control(pipe.unet, controller)

#             g = torch.Generator(device="cuda")
#             g.manual_seed(i)

#             images = pipe(prompt,
#                          num_images_per_prompt=1,
#                          num_inference_steps=30, 
#                           generator=g).images

#             images[0].save(
#                 './results-sdxl/rem_snoopy-3/{}/{}.png'.format(concept, tmp.format(concept)[:-1]+'_'+str(i))
#                 )
    



