# MSKDNP
## The article has been published and the training and distillation codes have been uploaded to GitHub.
Please cite：https://academic.oup.com/bib/article/26/5/bbaf466/8250822

## Server Status Notice（Updated on 4/16）

The main server has been restored, but it may still experience occasional instability.  
If you encounter any issues accessing the service, you can continue to use the temporary server at:

👉 https://ycclab.cuhk.edu.cn/mskdnp/

We apologize for any inconvenience this may cause and appreciate your understanding.


  

## 📌 Highlights

- 🔬 **Multi-Stage Knowledge Distillation**:
   A novel **multi-stage distillation** strategy is used to transfer classification knowledge from a high-capacity **teacher model** to a compact **student model**, achieving **over 98% parameter reduction** while preserving performance.
- ⚡ **Efficient Neuropeptide Prediction**:
   Designed for **low-resource environments**, enabling high-speed inference with **minimal computational cost**.
- 🧠 **Interpretability Aligned with Biology**:
   The model demonstrates strong **interpretability**, with attention regions aligning well with both **experimentally validated** and **computationally predicted** functional sites.
- 🌐 **Web Server Available**:
   A publicly accessible ****[web server (Click Here)](https://awi.cuhk.edu.cn/~biosequence/MSKDNP/index.php)**** incorporating protein language model knowledge is provided for broad neuropeptide **research and application use**.





<img src=".\figure\Interpretive.png" alt="Interpretive" style="zoom: 25%;" />





------

## 🚀 Usage

```
python3 main.py \
  --input_file data/proteins.fasta \
  --output_original results/features_320d.npy \
  --output_mapped results/features_1280d.npy \
  --output_predictions results/predictions.csv \
  --batch_size 32 \
  --make_prediction
```

------

## 🧾 Arguments

| Argument               | Type   | Required                   | Description                                                  |
| ---------------------- | ------ | -------------------------- | ------------------------------------------------------------ |
| `--input_file`         | `str`  | ✅ Yes                      | Path to the input FASTA file containing protein sequences.   |
| `--output_original`    | `str`  | ✅ Yes                      | Path to save the extracted 320-dimensional features (NumPy format). |
| `--output_mapped`      | `str`  | ✅ Yes                      | Path to save the mapped 1280-dimensional features (NumPy format). |
| `--output_predictions` | `str`  | ❌ Optional                 | Path to save classification results (CSV format). Required if `--make_prediction` is used. |
| `--batch_size`         | `int`  | ❌ Optional (default: `32`) | Batch size used during feature extraction.                   |
| `--make_prediction`    | `flag` | ❌ Optional                 | If set, enables prediction with the clasiifier head and saves outputs to `--output_predictions`. |

------

## 📦 Output Files

- `features_320d.npy`: Extracted original features with 320 dimensions.
- `features_1280d.npy`: Mapped feature representations with 1280 dimensions.
- predictions.csv: Predicted labels or scores if classification is performed.
