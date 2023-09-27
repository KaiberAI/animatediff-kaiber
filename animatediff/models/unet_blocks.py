# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_blocks.py

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from animatediff.models.attention import Transformer3DModel
from animatediff.models.motion_module import get_motion_module
from animatediff.models.resnet import Downsample3D, ResnetBlock3D, Upsample3D


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    unet_use_cross_frame_attention=None,
    unet_use_temporal_attention=None,
    use_inflated_groupnorm=None,
    use_motion_module=None,
    motion_module_type=None,
    motion_module_kwargs=None,
    trained_initial_frames=False,
):
    down_block_type = (
        down_block_type[7:]
        if down_block_type.startswith("UNetRes")
        else down_block_type
    )
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
            trained_initial_frames=trained_initial_frames
        )
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnDownBlock3D"
            )
        return CrossAttnDownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            unet_use_cross_frame_attention=unet_use_cross_frame_attention,
            unet_use_temporal_attention=unet_use_temporal_attention,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
            trained_initial_frames=trained_initial_frames
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    unet_use_cross_frame_attention=None,
    unet_use_temporal_attention=None,
    use_inflated_groupnorm=None,
    use_motion_module=None,
    motion_module_type=None,
    motion_module_kwargs=None,
    trained_initial_frames=False,
):
    up_block_type = (
        up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    )
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
            trained_initial_frames=trained_initial_frames
        )
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnUpBlock3D"
            )
        return CrossAttnUpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            unet_use_cross_frame_attention=unet_use_cross_frame_attention,
            unet_use_temporal_attention=unet_use_temporal_attention,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
            trained_initial_frames=trained_initial_frames
        )
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
        trained_initial_frames=False
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )

        # there is always at least one resnet
        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                use_inflated_groupnorm=use_inflated_groupnorm,
            )
        ]
        attentions = []
        motion_modules = []

        for _ in range(num_layers):
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )
            motion_module_kwargs_copy = motion_module_kwargs.copy()

            if trained_initial_frames:
                motion_module_kwargs_copy['train_init_channels_setup'] = True

            motion_modules.append(
                get_motion_module(
                    in_channels=in_channels if not trained_initial_frames else in_channels * 2,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs_copy,
                )
                if use_motion_module
                else None
            )
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        interpolate_pos_emb=0,
        attention_mask_last_n_frames=0,
        trained_initial_frames_input: torch.Tensor = None,
    ) -> torch.FloatTensor:
        hidden_states = self.resnets[0](hidden_states, temb)

        if trained_initial_frames_input != None:
            trained_initial_frames_input = self.resnets[0](trained_initial_frames_input, temb)

        for attn, resnet, motion_module in zip(
            self.attentions, self.resnets[1:], self.motion_modules
        ):
            hidden_states = attn(
                hidden_states.requires_grad_(),
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

            if trained_initial_frames_input != None:
                trained_initial_frames_input = attn(
                    trained_initial_frames_input,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            
            # add motion module
            hidden_states = (
                motion_module(
                    torch.cat((hidden_states, trained_initial_frames_input), 1) if trained_initial_frames_input != None else hidden_states,
                    temb,
                    encoder_hidden_states=encoder_hidden_states,
                    interpolate_pos_emb=interpolate_pos_emb,
                    attention_mask_last_n_frames=attention_mask_last_n_frames,
                )
                if motion_module is not None
                else hidden_states
            )

            hidden_states = resnet(hidden_states, temb)

            if trained_initial_frames_input != None:
                trained_initial_frames_input = resnet(trained_initial_frames_input, temb)

        return hidden_states, trained_initial_frames_input


class CrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
        trained_initial_frames=None,
    ):
        super().__init__()
        resnets = []
        attentions = []
        motion_modules = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    num_attention_heads=attn_num_head_channels,
                    attention_head_dim=out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )
            motion_module_kwargs_copy = motion_module_kwargs.copy()

            if trained_initial_frames:
                motion_module_kwargs_copy['train_init_channels_setup'] = True

            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels if not trained_initial_frames else out_channels * 2,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs_copy,
                )
                if use_motion_module
                else None
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        interpolate_pos_emb=0,
        attention_mask_last_n_frames=0,
        trained_initial_frames_input: torch.Tensor = None,
    ) -> torch.FloatTensor:
        output_states = ()
        output_initial_frames_states = ()

        for resnet, attn, motion_module in zip(self.resnets, self.attentions, self.motion_modules):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states.requires_grad_(),
                    encoder_hidden_states,
                )[0]
                
                if trained_initial_frames_input != None:
                    trained_initial_frames_input = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), trained_initial_frames_input, temb
                    )
                    trained_initial_frames_input = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        trained_initial_frames_input,
                        encoder_hidden_states,
                    )[0]
                
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        torch.cat((hidden_states.requires_grad_(), trained_initial_frames_input), 1) if trained_initial_frames_input != None else hidden_states.requires_grad_(),
                        temb,
                        encoder_hidden_states,
                        attention_mask_last_n_frames,
                        interpolate_pos_emb,
                    )

            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states.requires_grad_(),
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                
                if trained_initial_frames_input != None:
                    trained_initial_frames_input = resnet(trained_initial_frames_input, temb)
                    trained_initial_frames_input = attn(
                    trained_initial_frames_input,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                
                # add motion module
                hidden_states = (
                    motion_module(
                        torch.cat((hidden_states, trained_initial_frames_input), 1) if trained_initial_frames_input != None else hidden_states,
                        temb,
                        encoder_hidden_states=encoder_hidden_states,
                        interpolate_pos_emb=interpolate_pos_emb,
                        attention_mask_last_n_frames=attention_mask_last_n_frames,
                    )
                    if motion_module is not None
                    else hidden_states
                )

            output_states = output_states + (hidden_states,)
            output_initial_frames_states = output_initial_frames_states + (trained_initial_frames_input,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
                trained_initial_frames_input = downsampler(trained_initial_frames_input) if trained_initial_frames_input != None else None

            output_states = output_states + (hidden_states,)
            output_initial_frames_states = output_initial_frames_states + (trained_initial_frames_input,)

        return hidden_states, output_states, trained_initial_frames_input, output_initial_frames_states


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
        attention_mask_last_n_frames=0,
        trained_initial_frames=False
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )
            motion_module_kwargs_copy = motion_module_kwargs.copy()

            if trained_initial_frames:
                motion_module_kwargs_copy['train_init_channels_setup'] = True

            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels if not trained_initial_frames else out_channels * 2,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs_copy,
                )
                if use_motion_module
                else None
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        interpolate_pos_emb=0,
        attention_mask_last_n_frames=0,
        trained_initial_frames_input: torch.Tensor = None,
    ):
        output_states = ()
        output_initial_frames_states = ()

        for resnet, motion_module in zip(self.resnets, self.motion_modules):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )

                if trained_initial_frames_input != None:
                    trained_initial_frames_input = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), trained_initial_frames_input, temb
                    )
                
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        torch.cat((hidden_states.requires_grad_(), trained_initial_frames_input), 1) if trained_initial_frames_input != None else hidden_states.requires_grad_(),
                        temb,
                        encoder_hidden_states,
                        attention_mask_last_n_frames,
                        interpolate_pos_emb,
                    )
            else:
                hidden_states = resnet(hidden_states, temb)

                if trained_initial_frames_input != None:
                    trained_initial_frames_input = resnet(trained_initial_frames_input, temb)
                
                # add motion module
                hidden_states = (
                    motion_module(
                        torch.cat((hidden_states, trained_initial_frames_input), 1) if trained_initial_frames_input != None else hidden_states,
                        temb,
                        encoder_hidden_states=encoder_hidden_states,
                        interpolate_pos_emb=interpolate_pos_emb,
                        attention_mask_last_n_frames=attention_mask_last_n_frames,
                    )
                    if motion_module is not None
                    else hidden_states
                )

            output_states = output_states + (hidden_states,)
            output_initial_frames_states = output_initial_frames_states + (trained_initial_frames_input,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
                trained_initial_frames_input = downsampler(trained_initial_frames_input) if trained_initial_frames_input != None else None

            output_states = output_states + (hidden_states,)
            output_initial_frames_states = output_initial_frames_states + (trained_initial_frames_input,)

        return hidden_states, output_states, trained_initial_frames_input, output_initial_frames_states


class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
        trained_initial_frames=False
    ):
        super().__init__()
        resnets = []
        attentions = []
        motion_modules = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )
            motion_module_kwargs_copy = motion_module_kwargs.copy()

            if trained_initial_frames:
                motion_module_kwargs_copy['train_init_channels_setup'] = True

            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels if not trained_initial_frames else out_channels * 2,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs_copy,
                )
                if use_motion_module
                else None
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample3D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        interpolate_pos_emb=0,
        attention_mask_last_n_frames=0,
        trained_initial_frames_input: torch.Tensor = None,
        res_hidden_states_tuple_initial_frames: Tuple[torch.FloatTensor, ...] = None,
    ):
        for resnet, attn, motion_module in zip(
            self.resnets, self.attentions, self.motion_modules
        ):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # pop res hidden states for initial frames
            if trained_initial_frames_input != None:
                res_hidden_states_initial_frames = res_hidden_states_tuple_initial_frames[-1]
                res_hidden_states_tuple_initial_frames = res_hidden_states_tuple_initial_frames[:-1]
                trained_initial_frames_input = torch.cat([trained_initial_frames_input, res_hidden_states_initial_frames], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                )[0]

                if trained_initial_frames_input != None:
                    trained_initial_frames_input = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), trained_initial_frames_input, temb
                    )
                    trained_initial_frames_input = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        trained_initial_frames_input,
                        encoder_hidden_states,
                    )[0]
                
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        torch.cat((hidden_states.requires_grad_(), trained_initial_frames_input), 1) if trained_initial_frames_input != None else hidden_states.requires_grad_(),
                        temb,
                        encoder_hidden_states,
                        attention_mask_last_n_frames,
                        interpolate_pos_emb,
                    )

            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states.requires_grad_(),
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                if trained_initial_frames_input != None:
                    trained_initial_frames_input = resnet(trained_initial_frames_input, temb)
                    trained_initial_frames_input = attn(
                        trained_initial_frames_input,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                
                # add motion module
                hidden_states = (
                    motion_module(
                        torch.cat((hidden_states, trained_initial_frames_input), 1) if trained_initial_frames_input != None else hidden_states,
                        temb,
                        encoder_hidden_states=encoder_hidden_states,
                        interpolate_pos_emb=interpolate_pos_emb,
                        attention_mask_last_n_frames=attention_mask_last_n_frames,
                    )
                    if motion_module is not None
                    else hidden_states
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
                trained_initial_frames_input = upsampler(trained_initial_frames_input, upsample_size) if trained_initial_frames_input != None else None

        return hidden_states, trained_initial_frames_input

class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
        trained_initial_frames=False
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )

            motion_module_kwargs_copy = motion_module_kwargs.copy()

            if trained_initial_frames:
                motion_module_kwargs_copy['train_init_channels_setup'] = True

            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels if not trained_initial_frames else out_channels * 2,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs_copy,
                )
                if use_motion_module
                else None
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample3D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        upsample_size=None,
        encoder_hidden_states=None,
        interpolate_pos_emb=0,
        attention_mask_last_n_frames=0,
        trained_initial_frames_input: torch.Tensor = None,
        res_hidden_states_tuple_initial_frames: Tuple[torch.FloatTensor, ...] = None,
    ):
        for resnet, motion_module in zip(self.resnets, self.motion_modules):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if trained_initial_frames_input != None:
                res_hidden_states_initial_frames = res_hidden_states_tuple_initial_frames[-1]
                res_hidden_states_tuple_initial_frames = res_hidden_states_tuple_initial_frames[:-1]
                trained_initial_frames_input = torch.cat([trained_initial_frames_input, res_hidden_states_initial_frames], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )

                if trained_initial_frames_input != None:
                    trained_initial_frames_input = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), trained_initial_frames_input, temb
                    )
                
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        torch.cat((hidden_states.requires_grad_(), trained_initial_frames_input), 1) if trained_initial_frames_input != None else hidden_states.requires_grad_(),
                        temb,
                        encoder_hidden_states,
                        attention_mask_last_n_frames,
                        interpolate_pos_emb,
                    )
            else:
                hidden_states = resnet(hidden_states, temb)

                if trained_initial_frames_input != None:
                    trained_initial_frames_input = resnet(trained_initial_frames_input, temb)

                # add motion module
                hidden_states = (
                    motion_module(
                        torch.cat((hidden_states, trained_initial_frames_input), 1) if trained_initial_frames_input != None else hidden_states,
                        temb,
                        encoder_hidden_states=encoder_hidden_states,
                        interpolate_pos_emb=interpolate_pos_emb,
                        attention_mask_last_n_frames=attention_mask_last_n_frames,
                    )
                    if motion_module is not None
                    else hidden_states
                )


        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
                trained_initial_frames_input = upsampler(trained_initial_frames_input, upsample_size) if trained_initial_frames_input != None else None

        return hidden_states, trained_initial_frames_input
