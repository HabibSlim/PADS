MODEL_MAP = [
    # {
    #     "model_name": "mlp_mapper_bert_l1__256",
    #     "checkpoint": "mlp_mapper_bert_l1__256/checkpoint-50.pth",
    #     "method_code": r"""\textsc{DirectGen}_{\textsc{Linear}}""",
    #     "is_decoupled": False,
    # },
    {
        "model_name": "mlp_mapper_bert_bneck_1024_pcae_cpl",
        "checkpoint": "mlp_mapper_bert_bneck_1024_pcae__fine_chained_cpl/checkpoint-59.pth",
        "method_code": r"""\textsc{LateFusion}_{1024}""",
        "is_decoupled": False,
    },
    {
        "model_name": "mlp_mapper_bert_bneck_512_pcae_cpl",
        "checkpoint": "mlp_mapper_bert_bneck_512_pcae__fine_cpl__chained/checkpoint-59.pth",
        "method_code": r"""\textsc{LateFusion}_{512}""",
        "is_decoupled": False,
    },
    {
        "model_name": "mlp_mapper_bert_bneck_256_pcae_cpl",
        "checkpoint": "mlp_mapper_bert_bneck_256_pcae__fine_cpl__chained/checkpoint-59.pth",
        "method_code": r"""\textsc{LateFusion}_{256}""",
        "is_decoupled": False,
    },
    {
        "model_name": "mlp_mapper_bert_l8_pcae_cpl",
        "checkpoint": "mlp_mapper_bert_l8_pcae__fine_cpl__chained/checkpoint-59.pth",
        "method_code": r"""\textsc{Ours}_{512 \times 8}""",
        "is_decoupled": False,
    },
    {
        "model_name": "mlp_mapper_bert_l4_pcae_cpl",
        "checkpoint": "mlp_mapper_bert_l4_pcae__fine_cpl__chained/checkpoint-59.pth",
        "method_code": r"""\textsc{Ours}_{512 \times 4}""",
        "is_decoupled": False,
    },
    {
        "model_name": "mlp_mapper_bert_bneck_1024_pcae",
        "checkpoint": "mlp_mapper_bert_bneck_1024_pcae__fine_chained/checkpoint-59.pth",
        "method_code": r"""\textsc{LateFusion}_{1024}""",
        "is_decoupled": True,
    },
    {
        "model_name": "mlp_mapper_bert_bneck_512_pcae",
        "checkpoint": "mlp_mapper_bert_bneck_512_pcae__fine_chained/checkpoint-59.pth",
        "method_code": r"""\textsc{LateFusion}_{512}""",
        "is_decoupled": True,
    },
    {
        "model_name": "mlp_mapper_bert_bneck_256_pcae",
        "checkpoint": "mlp_mapper_bert_bneck_256_pcae__fine_chained/checkpoint-59.pth",
        "method_code": r"""\textsc{LateFusion}_{256}""",
        "is_decoupled": True,
    },
    {
        "model_name": "mlp_mapper_bert_l8_pcae",
        "checkpoint": "mlp_mapper_bert_l8_pcae__fine_chained/checkpoint-59.pth",
        "method_code": r"""\textsc{Ours}_{512 \times 8}""",
        "is_decoupled": True,
    },
    {
        "model_name": "mlp_mapper_bert_l4_pcae",
        "checkpoint": "mlp_mapper_bert_l4_pcae__fine_chained/checkpoint-59.pth",
        "method_code": r"""\textsc{Ours}_{512 \times 4}""",
        "is_decoupled": True,
    },
]

MODEL_MAP_IMNET = [
    # {
    #     "model_name": "mlp_mapper_bert_l1__256",
    #     "checkpoint": "mlp_mapper_bert_l1__256/checkpoint-50.pth",
    #     "method_code": r"""\textsc{DirectGen}_{\textsc{Linear}}""",
    #     "is_decoupled": False,
    # },
    {
        "model_name": "mlp_mapper_bert_bneck_1024_pcae_cpl",
        "checkpoint": "mlp_mapper_bert_bneck_1024_imnet__fine_chained_cpl/checkpoint-59.pth",
        "method_code": r"""\textsc{LateFusion}_{1024}""",
        "is_decoupled": False,
    },
    {
        "model_name": "mlp_mapper_bert_bneck_512_pcae_cpl",
        "checkpoint": "mlp_mapper_bert_bneck_512_imnet__fine_chained_cpl/checkpoint-59.pth",
        "method_code": r"""\textsc{LateFusion}_{512}""",
        "is_decoupled": False,
    },
    {
        "model_name": "mlp_mapper_bert_bneck_256_pcae_cpl",
        "checkpoint": "mlp_mapper_bert_bneck_256_imnet__fine_chained_cpl/checkpoint-59.pth",
        "method_code": r"""\textsc{LateFusion}_{256}""",
        "is_decoupled": False,
    },
    {
        "model_name": "mlp_mapper_bert_l8_pcae_cpl",
        "checkpoint": "mlp_mapper_bert_l8_imnet__fine_chained_cpl/checkpoint-59.pth",
        "method_code": r"""\textsc{Ours}_{512 \times 8}""",
        "is_decoupled": False,
    },
    {
        "model_name": "mlp_mapper_bert_l4_pcae_cpl",
        "checkpoint": "mlp_mapper_bert_l4_imnet__fine_chained_cpl/checkpoint-59.pth",
        "method_code": r"""\textsc{Ours}_{512 \times 4}""",
        "is_decoupled": False,
    },
    {
        "model_name": "mlp_mapper_bert_bneck_1024_pcae",
        "checkpoint": "mlp_mapper_bert_bneck_1024_imnet__fine_chained/checkpoint-59.pth",
        "method_code": r"""\textsc{LateFusion}_{1024}""",
        "is_decoupled": True,
    },
    {
        "model_name": "mlp_mapper_bert_bneck_512_pcae",
        "checkpoint": "mlp_mapper_bert_bneck_512_imnet__fine_chained/checkpoint-59.pth",
        "method_code": r"""\textsc{LateFusion}_{512}""",
        "is_decoupled": True,
    },
    {
        "model_name": "mlp_mapper_bert_bneck_256_pcae",
        "checkpoint": "mlp_mapper_bert_bneck_256_imnet__fine_chained/checkpoint-59.pth",
        "method_code": r"""\textsc{LateFusion}_{256}""",
        "is_decoupled": True,
    },
    {
        "model_name": "mlp_mapper_bert_l8_pcae",
        "checkpoint": "mlp_mapper_bert_l8_imnet__fine_chained/checkpoint-59.pth",
        "method_code": r"""\textsc{Ours}_{512 \times 8}""",
        "is_decoupled": True,
    },
    {
        "model_name": "mlp_mapper_bert_l4_pcae",
        "checkpoint": "mlp_mapper_bert_l4_imnet__fine_chained/checkpoint-59.pth",
        "method_code": r"""\textsc{Ours}_{512 \times 4}""",
        "is_decoupled": True,
    },
]
