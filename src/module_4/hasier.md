## Push notifications project

El objetivo de este notebook es intentar crear un modelo que pueda predecir si el usuario va a comprarar el producto o no. Se va a empezar con un modelo simple y se va ir complicando el modelo con la intencion de mejorarlo.
1. - Basaline.
2. - Modelos lineales.
3. - Modelos no lineales.



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


from typing import Tuple 
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc, average_precision_score, log_loss
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

```


```python
base_path = Path("../../../datos_originales").resolve()
dataset_path = base_path / Path("feature_frame.csv")
```


```python
df = pd.read_csv(dataset_path)
```


```python
df["order_date"] = pd.to_datetime(df["order_date"]).apply(lambda x: x.date())
df["created_at"] = pd.to_datetime(df["created_at"])
```


```python
df.sort_values(by="order_id").head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.276180</td>
    </tr>
    <tr>
      <th>1957119</th>
      <td>34173018275972</td>
      <td>coffee</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>26.5</td>
      <td>25.529657</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>24.363052</td>
    </tr>
    <tr>
      <th>68955</th>
      <td>33826466758788</td>
      <td>tea</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>41.0</td>
      <td>30.651733</td>
      <td>30.0</td>
      <td>33.0</td>
      <td>26.060328</td>
    </tr>
    <tr>
      <th>1960565</th>
      <td>33803541512324</td>
      <td>washingliquidgel</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>37.0</td>
      <td>25.327905</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>26.445904</td>
    </tr>
    <tr>
      <th>1971726</th>
      <td>33667275227268</td>
      <td>deodorant</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>8.0</td>
      <td>27.693045</td>
      <td>30.0</td>
      <td>36.0</td>
      <td>27.978627</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
print(df.shape[0])
print(df.columns)
```

    2880549
    Index(['variant_id', 'product_type', 'order_id', 'user_id', 'created_at',
           'order_date', 'user_order_seq', 'outcome', 'ordered_before',
           'abandoned_before', 'active_snoozed', 'set_as_regular',
           'normalised_price', 'discount_pct', 'vendor', 'global_popularity',
           'count_adults', 'count_children', 'count_babies', 'count_pets',
           'people_ex_baby', 'days_since_purchase_variant_id',
           'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
           'days_since_purchase_product_type', 'avg_days_to_buy_product_type',
           'std_days_to_buy_product_type'],
          dtype='object')


### Project

Se va a intentar entrenar un modelo. El EDA se hizo previamente y se vio que era un problema muy desvalanceado. Esto va ser importante a la hora de hacer el modelo y de enternder las diferentes graficas.
Para poder entrenar el modelo se van a seguir estos pasos:
1. Baseline
2. Modelo lineal
3. Modelo no lineal

Desde un modelo mas simple a un modelo mas complicado. Nos vamos  a quedar con el mejor resultado.



### Data splits

En esta sección, dividiremos los datos en 3 grupos:

1. Entrenamiento: para ajustar nuestros modelos.
2. Validación: para seleccionar el mejor modelo y la mejor hiperparametrización. Además, para realizar la calibración.
3. Prueba: para evaluar el rendimiento del modelo final.

Siguiendo el documento de requisitos del producto (PRD), solo estamos interesados en los pedidos que tengan al menos 5 artículos en la cesta.


```python
df_outcome_true = df.loc[df["outcome"] == 1]
```


```python
num_filas = df.shape[0]
num_filas_outcome_true = df_outcome_true.shape[0]
print(f"El dataset tiene {num_filas} filas.")
print(f"El dataset tiene {num_filas_outcome_true} filas con outcome igual a 1.")
```

    El dataset tiene 2880549 filas.
    El dataset tiene 33232 filas con outcome igual a 1.



```python
variant_counts = df_outcome_true["variant_id"].value_counts()
print(variant_counts)
```

    variant_id
    34081589887108    1106
    34370915041412     331
    33973249081476     297
    33667282436228     288
    34370361229444     270
                      ... 
    34535159758980       1
    34535159726212       1
    34529809170564       1
    34529808875652       1
    33667293511812       1
    Name: count, Length: 913, dtype: int64



```python
num_unique_variants = df_outcome_true["variant_id"].nunique()
print(f"El número de variant_id diferentes es: {num_unique_variants}")

```

    El número de variant_id diferentes es: 913



```python
order_counts = df_outcome_true["order_id"].value_counts()
print(order_counts)
```

    order_id
    2907881767044    78
    2887290192004    53
    2816464388228    53
    2888290697348    45
    3639453843588    44
                     ..
    2906542211204     1
    2906584318084     1
    2906635567236     1
    2906639270020     1
    2923260051588     1
    Name: count, Length: 3427, dtype: int64


@gscharly

Aqui porque hacemos variante_id.unique()?
No se podria hacer hacer como lo he puesto abajo? Porque te dice q nos interesan cestas de 5 o mas productos pero no con variante_id diferente no?



```python
basket_size = df_outcome_true.groupby("order_id")["variant_id"].nunique()
order_ids_basket_5plus = basket_size[basket_size >= 5].index
df_basket5 = df[df["order_id"].isin(order_ids_basket_5plus)]


#basket_size = df_outcome_true.groupby("order_id")["variant_id"].count()
#order_ids_basket_5plus = basket_size[basket_size >= 5].index
#df_basket5 = df[df["order_id"].isin(order_ids_basket_5plus)]
```


```python
# Primera forma: Contando variantes únicas
basket_size_unique = df_outcome_true.groupby("order_id")["variant_id"].nunique()
order_ids_basket_5plus_unique = basket_size_unique[basket_size_unique >= 5].index
df_basket5_unique = df[df["order_id"].isin(order_ids_basket_5plus_unique)]

# Segunda forma: Contando todas las variantes
basket_size_total = df_outcome_true.groupby("order_id")["variant_id"].count()
order_ids_basket_5plus_total = basket_size_total[basket_size_total >= 5].index
df_basket5_total = df[df["order_id"].isin(order_ids_basket_5plus_total)]

# Comparación de los resultados
print("Número de órdenes con 5 o más variantes únicas:", len(order_ids_basket_5plus_unique))
print("Número de órdenes con 5 o más variantes totales:", len(order_ids_basket_5plus_total))
```

    Número de órdenes con 5 o más variantes únicas: 2603
    Número de órdenes con 5 o más variantes totales: 2603


Da el mismo numero, pero no entiendo el porque se hace nunique().


```python
daily_orders = df_basket5.groupby("order_date")["order_id"].nunique()
cumulative_orders = (daily_orders.cumsum() / daily_orders.sum()) * 100
```


```python

fig, ax = plt.subplots()
lns1 = ax.plot(daily_orders, label="daily orders")
axi = ax.twinx()
lns2 = axi.plot(cumulative_orders, color='k', label="cumulative orders")
ax.set_ylabel("Daily Orders")
axi.set_ylabel("Cumulative orders in the data set")
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

```




    <matplotlib.legend.Legend at 0x7f9845651f50>




    
![png](hasier_files/hasier_22_1.png)
    


Con esta grafica podemos ver la evolucion de los pedidos. 


```python

train_val_cutoff = cumulative_orders[cumulative_orders <= 70].index[-1]
val_test_cutoff = cumulative_orders[cumulative_orders <= 98].index[-1]


print(f"Train since: {cumulative_orders.index.min()}")
print(f"Train until: {train_val_cutoff}")
print(f"Val until: {val_test_cutoff}")
print(f"Test until: {cumulative_orders.index.max()}")

```

    Train since: 2020-10-05
    Train until: 2021-02-04
    Val until: 2021-02-28
    Test until: 2021-03-03



```python
train_df = df_basket5.loc[df_basket5["order_date"] <= train_val_cutoff]
val_df = df_basket5.loc[
    (df_basket5["order_date"] > train_val_cutoff)
    & (df_basket5["order_date"] <= val_test_cutoff)
]
test_df = df_basket5.loc[df_basket5["order_date"] > val_test_cutoff]

item_popularity = train_df.groupby("variant_id").apply(
    lambda x: x.loc[x["outcome"] == 1, "order_id"].nunique()
)
item_popularity = item_popularity.sort_values(ascending=False)

X_train, y_train = (train_df.drop("outcome", axis=1), train_df.loc[:, "outcome"])
X_val, y_val = (val_df.drop("outcome", axis=1), val_df.loc[:, "outcome"])
X_test, y_test = (test_df.drop("outcome", axis=1), test_df.loc[:, "outcome"])
```

### Baseline


```python
top_training_items = (
    X_train.loc[y_train == 1]
    .groupby("variant_id")
    .apply(lambda x: x["order_id"].nunique())
)

top_training_items /= top_training_items.sum()

baseline_predictions = val_df.copy().reset_index(drop=True)
baseline_predictions.loc[:, "proba"] = (
    top_training_items.reindex(val_df["variant_id"].values)
    .fillna(0)
    .reset_index(drop=True)
)
```


```python
(
    baseline_push_precision,
    baseline_push_recall,
    baseline_push_pr_thresholds,
) = precision_recall_curve(
    baseline_predictions["outcome"], baseline_predictions["proba"]
)

(
    baseline_push_fpr, 
    baseline_push_tpr, 
    baseline_push_roc_thresholds
) = roc_curve(
    baseline_predictions["outcome"], baseline_predictions["proba"]
)

baseline_push_auc = roc_auc_score(
    baseline_predictions["outcome"], baseline_predictions["proba"]
)

baseline_push_ap = average_precision_score(
    baseline_predictions["outcome"], baseline_predictions["proba"]
)
```


```python
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

ax[0].plot(baseline_push_recall, baseline_push_precision)
ax[0].set_xlabel("recall")
ax[0].set_ylabel("precision")
ax[0].set_title(f"AP: {baseline_push_ap:.4f}")

ax[1].plot(baseline_push_fpr, baseline_push_tpr)
ax[1].set_xlabel("FPR")
ax[1].set_ylabel("TPR")
ax[1].set_title(f"AUC: {baseline_push_auc:.4f}")

plt.show()
```


    
![png](hasier_files/hasier_29_0.png)
    


Sabemos que es un problema desvalanceado y por la forma en las que se calculan las curvas, el precision recall curve nos va a dar mas informacion. Va ser mas representativa.

### First model


```python
print(train_df.columns)

```

    Index(['variant_id', 'product_type', 'order_id', 'user_id', 'created_at',
           'order_date', 'user_order_seq', 'outcome', 'ordered_before',
           'abandoned_before', 'active_snoozed', 'set_as_regular',
           'normalised_price', 'discount_pct', 'vendor', 'global_popularity',
           'count_adults', 'count_children', 'count_babies', 'count_pets',
           'people_ex_baby', 'days_since_purchase_variant_id',
           'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
           'days_since_purchase_product_type', 'avg_days_to_buy_product_type',
           'std_days_to_buy_product_type'],
          dtype='object')



```python
columns = [
    "ordered_before",
    "abandoned_before",
    "normalised_price",
    "set_as_regular",
    "active_snoozed",
    "discount_pct",
    "days_since_purchase_variant_id",
    "avg_days_to_buy_variant_id",
    "std_days_to_buy_variant_id",
    "days_since_purchase_product_type",
    "avg_days_to_buy_product_type",
    "std_days_to_buy_product_type",
    "global_popularity",
    "people_ex_baby",
    "count_adults",
    "count_children",
    "count_babies",
    "count_pets",
    "user_order_seq"
]
```


```python
def evaluate_configuration(
    clf,
    X_train,
    y_train,
    X_val,
    y_val,
    train_aucs_list,
    val_aucs_list,
    train_ce_list,
    val_ce_list,
    train_aps_list,
    val_aps_list
):
    val_preds = clf.predict_proba(X_val)[:, 1]
    train_preds = clf.predict_proba(X_train)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_preds)
    val_auc = roc_auc_score(y_val, val_preds)
    
    train_crossentropy = log_loss(y_train, train_preds)
    val_crossentropy = log_loss(y_val, val_preds)
    
    train_ap = average_precision_score(y_train, train_preds)
    val_ap = average_precision_score(y_val, val_preds)
    
    train_aucs_list.append(train_auc)
    val_aucs_list.append(val_auc)
    train_ce_list.append(train_crossentropy)
    val_ce_list.append(val_crossentropy)
    train_aps_list.append(train_ap)
    val_aps_list.append(val_ap)
```


```python
def plot_feature_importance(clf, columns):
    fig, ax = plt.subplots()
    fi = pd.DataFrame(
        list(zip(columns, clf.feature_importances_)), columns=["features", "importance"]
    ).sort_values(by="importance", ascending=True)
    fi.plot(kind="barh", x="features", y="importance", ax=ax)
    return fi, fig, ax
```


```python

corr = train_df[columns + ["outcome"]].corr()


mask = np.triu(np.ones_like(corr, dtype=bool))


fig, ax = plt.subplots(figsize=(11, 9))


cmap = sns.diverging_palette(230, 20, as_cmap=True)


sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=0.3,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5}
)

plt.show()
```


    
![png](hasier_files/hasier_36_0.png)
    


Se puede ver q algunas variables estan correladas.

### Linear models

Ridge y Lasso


```python

def train_and_evaluate_model(c, X_train, y_train, X_val, y_val, columns):
    if c is None:
        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty='l2', C=1.0)
        )
    else:
        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty='l2', C=c)
        )
    
    lr.fit(X_train[columns], y_train)
    
    train_preds = lr.predict_proba(X_train[columns])[:, 1]
    val_preds = lr.predict_proba(X_val[columns])[:, 1]
    
    train_auc = roc_auc_score(y_train, train_preds)
    val_auc = roc_auc_score(y_val, val_preds)
    
    train_crossentropy = log_loss(y_train, train_preds)
    val_crossentropy = log_loss(y_val, val_preds)
    
    train_ap = average_precision_score(y_train, train_preds)
    val_ap = average_precision_score(y_val, val_preds)
    
    return train_auc, val_auc, train_crossentropy, val_crossentropy, train_ap, val_ap
```


```python

lr_push_train_aucs = []
lr_push_val_aucs = []
lr_push_train_ce = []
lr_push_val_ce = []
lr_push_train_aps = []
lr_push_val_aps = []


cs = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100, 1e4, None]

for c in cs:
    train_auc, val_auc, train_ce, val_ce, train_ap, val_ap = train_and_evaluate_model(c, X_train, y_train, X_val, y_val, columns)
    
    lr_push_train_aucs.append(train_auc)
    lr_push_val_aucs.append(val_auc)
    lr_push_train_ce.append(train_ce)
    lr_push_val_ce.append(val_ce)
    lr_push_train_aps.append(train_ap)
    lr_push_val_aps.append(val_ap)

    print(
        f"C: {c} done with train auc: {train_auc:.4f} and val auc: {val_auc:.4f}, "
        f"AP train: {train_ap:.4f} and val: {val_ap:.4f}, "
        f"Cross entropy train: {train_ce:.4f} and val: {val_ce:.4f}"
    )
```

    C: 1e-08 done with train auc: 0.8247 and val auc: 0.8274, AP train: 0.1560 and val: 0.1553, Cross entropy train: 0.0782 and val: 0.0718
    C: 1e-06 done with train auc: 0.8254 and val auc: 0.8282, AP train: 0.1562 and val: 0.1555, Cross entropy train: 0.0762 and val: 0.0696
    C: 0.0001 done with train auc: 0.8142 and val auc: 0.8192, AP train: 0.1599 and val: 0.1547, Cross entropy train: 0.0665 and val: 0.0607
    C: 0.01 done with train auc: 0.8007 and val auc: 0.8080, AP train: 0.1581 and val: 0.1522, Cross entropy train: 0.0663 and val: 0.0606
    C: 1 done with train auc: 0.8003 and val auc: 0.8077, AP train: 0.1581 and val: 0.1521, Cross entropy train: 0.0663 and val: 0.0606
    C: 100 done with train auc: 0.8003 and val auc: 0.8077, AP train: 0.1581 and val: 0.1521, Cross entropy train: 0.0663 and val: 0.0606
    C: 10000.0 done with train auc: 0.8003 and val auc: 0.8077, AP train: 0.1581 and val: 0.1521, Cross entropy train: 0.0663 and val: 0.0606
    C: None done with train auc: 0.8003 and val auc: 0.8077, AP train: 0.1581 and val: 0.1521, Cross entropy train: 0.0663 and val: 0.0606



```python
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# AUC plot
ax = axes[0]
ax.plot(np.arange(len(cs)), lr_push_train_aucs, label='train auc')
ax.plot(np.arange(len(cs)), lr_push_val_aucs, label='val auc')
ax.hlines(
    baseline_push_auc, 0, len(cs) - 1, colors='k', linestyles='--', label='baseline'
)
ax.set_xticks(np.arange(len(cs)))
ax.set_xticklabels(cs)
ax.set_xlabel('C')
ax.set_ylabel('AUC')
ax.legend()
ax.grid()

# Average Precision (AP) plot
ax = axes[1]
ax.plot(np.arange(len(cs)), lr_push_train_aps, label='train ap')
ax.plot(np.arange(len(cs)), lr_push_val_aps, label='val ap')
ax.hlines(
    baseline_push_ap, 0, len(cs) - 1, colors='k', linestyles='--', label='baseline'
)
ax.set_xticks(np.arange(len(cs)))
ax.set_xticklabels(cs)
ax.set_xlabel('C')
ax.set_ylabel('AP')
ax.legend()
ax.grid()

plt.show()
```


    
![png](hasier_files/hasier_42_0.png)
    



```python
def create_and_train_logistic_regression_model(c, X_train, y_train, columns):
    if c is None:
        # Si c es None, se desactiva la regularización
        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty=None, solver='saga', max_iter=10000)
        )
    else:
        # Si c tiene un valor numérico, se utiliza regularización L1 con el valor de C dado
        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty='l1', C=c, solver='saga', max_iter=10000)
        )
    
    lr.fit(X_train[columns], y_train)
    return lr


```


```python
def evaluate_models_with_different_regularization(cs, X_train, y_train, X_val, y_val, columns):
    # Inicializar listas para almacenar los resultados
    train_aucs, val_aucs = [], []
    train_ce, val_ce = [], []
    train_aps, val_aps = [], []

    for c in cs:
        model = create_and_train_logistic_regression_model(c, X_train, y_train, columns)
        
        evaluate_configuration(
            model,
            X_train[columns],
            y_train,
            X_val[columns],
            y_val,
            train_aucs,
            val_aucs,
            train_ce,
            val_ce,
            train_aps,
            val_aps
        )

        print(
            f"C: {c} done with train auc: {train_aucs[-1]:.4f} and val auc: {val_aucs[-1]:.4f}, "
            f"AP train: {train_aps[-1]:.4f} and val: {val_aps[-1]:.4f}, "
            f"Cross entropy train: {train_ce[-1]:.4f} and val: {val_ce[-1]:.4f}"
        )
    
    return train_aucs, val_aucs, train_ce, val_ce, train_aps, val_aps

```


```python
# Lista de valores para la regularización
cs = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100, 1e4, None]

# Entrenar y evaluar el modelo con los valores de C dados
lr_push_train_aucs, lr_push_val_aucs, lr_push_train_ce, lr_push_val_ce, lr_push_train_aps, lr_push_val_aps = evaluate_models_with_different_regularization(
    cs, X_train, y_train, X_val, y_val, columns
)

```

    C: 1e-08 done with train auc: 0.5000 and val auc: 0.5000, AP train: 0.0151 and val: 0.0135, Cross entropy train: 0.0783 and val: 0.0718
    C: 1e-06 done with train auc: 0.5000 and val auc: 0.5000, AP train: 0.0151 and val: 0.0135, Cross entropy train: 0.0783 and val: 0.0718
    C: 0.0001 done with train auc: 0.8330 and val auc: 0.8324, AP train: 0.1615 and val: 0.1441, Cross entropy train: 0.0678 and val: 0.0624
    C: 0.01 done with train auc: 0.8021 and val auc: 0.8093, AP train: 0.1583 and val: 0.1526, Cross entropy train: 0.0663 and val: 0.0606
    C: 1 done with train auc: 0.8004 and val auc: 0.8078, AP train: 0.1581 and val: 0.1522, Cross entropy train: 0.0663 and val: 0.0606
    C: 100 done with train auc: 0.8004 and val auc: 0.8078, AP train: 0.1581 and val: 0.1522, Cross entropy train: 0.0663 and val: 0.0606
    C: 10000.0 done with train auc: 0.8004 and val auc: 0.8078, AP train: 0.1581 and val: 0.1522, Cross entropy train: 0.0663 and val: 0.0606
    C: None done with train auc: 0.8004 and val auc: 0.8078, AP train: 0.1581 and val: 0.1522, Cross entropy train: 0.0663 and val: 0.0606



```python
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# AUC plot
ax = axes[0]
ax.plot(np.arange(len(cs)), lr_push_train_aucs, label='train auc')
ax.plot(np.arange(len(cs)), lr_push_val_aucs, label='val auc')
ax.hlines(baseline_push_auc, 0, len(cs) - 1, colors='k', linestyles='--', label='baseline')
ax.set_xticks(np.arange(len(cs)))
ax.set_xticklabels(cs)
ax.set_xlabel('C')
ax.set_ylabel('AUC')
ax.legend()
ax.grid()

# Average Precision (AP) plot
ax = axes[1]
ax.plot(np.arange(len(cs)), lr_push_train_aps, label='train ap')
ax.plot(np.arange(len(cs)), lr_push_val_aps, label='val ap')
ax.hlines(baseline_push_ap, 0, len(cs) - 1, colors='k', linestyles='--', label='baseline')
ax.set_xticks(np.arange(len(cs)))
ax.set_xticklabels(cs)
ax.set_xlabel('C')
ax.set_ylabel('AP')
ax.legend()
ax.grid()

plt.show()
```


    
![png](hasier_files/hasier_46_0.png)
    



```python
# Modelo con L2
lr = Pipeline(
    [
        ("standard_scaler", StandardScaler()),
        ("lr", LogisticRegression(penalty="l2", C=1e-6)),
    ]
)

lr.fit(X_train[columns], y_train)
lr_coeff_l2 = pd.DataFrame(
    {
        "features": columns,
        "importance": np.abs(lr.named_steps["lr"].coef_[0]),
        "regularisation": ["l2"] * len(columns),
    }
)
lr_coeff_l2 = lr_coeff_l2.sort_values("importance", ascending=True)

# Modelo con L1
lr = Pipeline(
    [
        ("standard_scaler", StandardScaler()),
        ("lr", LogisticRegression(penalty="l1", C=1e-4, solver="saga")),
    ]
)

lr.fit(X_train[columns], y_train)
lr_coeff_l1 = pd.DataFrame(
    {
        "features": columns,
        "importance": np.abs(lr.named_steps["lr"].coef_[0]),
        "regularisation": ["l1"] * len(columns),
    }
)
lr_coeff_l1 = lr_coeff_l1.sort_values("importance", ascending=True)
```


```python
# Concatenar los coeficientes de los modelos L2 y L1
lr_coeffs = pd.concat([lr_coeff_l2, lr_coeff_l1])

# Convertir la columna 'features' a categórica
lr_coeffs['features'] = pd.Categorical(lr_coeffs['features'])

# Ordenar los coeficientes por importancia
lr_coeffs = lr_coeffs.sort_values(by='importance')

# Ordenar las columnas por importancia descendente
order_columns = lr_coeff_l2.sort_values(by='importance', ascending=False)['features']

# Crear el gráfico de barras
sns.barplot(
    data=lr_coeffs,
    x='importance',
    y='features',
    hue='regularisation',
    order=order_columns,
)
```




    <Axes: xlabel='importance', ylabel='features'>




    
![png](hasier_files/hasier_48_1.png)
    



```python
# Seleccionar las características con importancia mayor a 0 en L1
l1_features = lr_coeff_l1.loc[lr_coeff_l1['importance'] > 0, 'features'].tolist()

# Crear el pipeline
lr = Pipeline(
    [
        ("standard_scaler", StandardScaler()),
        ("lr", LogisticRegression(penalty="l1", C=1e-4, solver="saga")),
    ]
)

# Ajustar el modelo con las características seleccionadas
lr.fit(X_train[l1_features], y_train)

# Predicciones
val_preds = lr.predict_proba(X_val[l1_features])[:, 1]
train_preds = lr.predict_proba(X_train[l1_features])[:, 1]

# Calcular las métricas
train_auc = roc_auc_score(y_train, train_preds)
val_auc = roc_auc_score(y_val, val_preds)

train_ap = average_precision_score(y_train, train_preds)
val_ap = average_precision_score(y_val, val_preds)

# Imprimir las métricas
print(f"Train AUC: {train_auc:.4f}")
print(f"Val AUC: {val_auc:.4f}")

print(f"Train AP: {train_ap:.4f}")
print(f"Val AP: {val_ap:.4f}")
```

    Train AUC: 0.8330
    Val AUC: 0.8324
    Train AP: 0.1615
    Val AP: 0.1441



```python
l2_features = lr_coeff_l2.loc[lr_coeff_l2['importance'] > 0, 'features'].tolist()
lr = Pipeline(
    [
        ("standard_scaler", StandardScaler()),
        ("lr", LogisticRegression(penalty="l2", C=1e-6)),
    ]
)

lr.fit(X_train[l2_features], y_train)
val_preds = lr.predict_proba(X_val[l2_features])[:, 1]
train_preds = lr.predict_proba(X_train[l2_features])[:, 1]

train_auc = roc_auc_score(y_train, train_preds)
val_auc = roc_auc_score(y_val, val_preds)

train_ap = average_precision_score(y_train, train_preds)
val_ap = average_precision_score(y_val, val_preds)

print(f"Train AUC: {train_auc:.4f}")
print(f"Val AUC: {val_auc:.4f}")

print(f"Train AP: {train_ap:.4f}")
print(f"Val AP: {val_ap:.4f}")
```

    Train AUC: 0.8254
    Val AUC: 0.8282
    Train AP: 0.1562
    Val AP: 0.1555


### Non linear models

- Ahora buscamos mejorar la complejidad de nuestro modelo para intentar aumentar su rendimiento.
- Exploraremos algunos  modelos no lineales para ver su rendimiento.

#### Random forest


```python
n_trees_grid = [5, 25, 50, 100]

rf_push_train_aucs = []
rf_push_val_aucs = []

rf_push_train_ce = []
rf_push_val_ce = []

rf_push_train_aps = []
rf_push_val_aps = []

#no hace falta escalar los datos en random forest
for n_trees in n_trees_grid:
    rf = RandomForestClassifier(n_trees)
    rf.fit(X_train[columns], y_train)
    evaluate_configuration(
        rf,
        X_train[columns],
        y_train,
        X_val[columns],
        y_val,
        rf_push_train_aucs,
        rf_push_val_aucs,
        rf_push_train_ce,
        rf_push_val_ce,
        rf_push_train_aps,
        rf_push_val_aps
    )
    print(
        f"# Trees: {n_trees} done with train auc: {rf_push_train_aucs[-1]:.4f} and val auc: {rf_push_val_aucs[-1]:.4f}, "
        f"AP train: {rf_push_train_aps[-1]:.4f} and val: {rf_push_val_aps[-1]:.4f}, "
        f"Cross entropy train: {rf_push_train_ce[-1]:.4f} and val: {rf_push_val_ce[-1]:.4f}"
    )
```

    # Trees: 5 done with train auc: 0.9906 and val auc: 0.6479, AP train: 0.7705 and val: 0.0736, Cross entropy train: 0.0246 and val: 0.3440
    # Trees: 25 done with train auc: 0.9952 and val auc: 0.7123, AP train: 0.8521 and val: 0.1117, Cross entropy train: 0.0209 and val: 0.2513
    # Trees: 50 done with train auc: 0.9955 and val auc: 0.7327, AP train: 0.8596 and val: 0.1154, Cross entropy train: 0.0208 and val: 0.2171
    # Trees: 100 done with train auc: 0.9957 and val auc: 0.7505, AP train: 0.8626 and val: 0.1202, Cross entropy train: 0.0207 and val: 0.1852



```python
rf = RandomForestClassifier(100)
rf.fit(X_train[columns], y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;RandomForestClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier()</pre></div> </div></div></div></div>




```python
rf_fi, fig, ax = plot_feature_importance(rf, columns)
```


    
![png](hasier_files/hasier_56_0.png)
    


quitamos los features menos importantes. Nos quedamos con los diez mas importantes.


```python
filtered_columns = rf_fi["features"].iloc[:10]
```


```python
n_trees_grid = [5, 25, 50, 100]

rf_push_train_aucs = []
rf_push_val_aucs = []

rf_push_train_ce = []
rf_push_val_ce = []

rf_push_train_aps = []
rf_push_val_aps = []

columns_to_use = filtered_columns

for n_trees in n_trees_grid:
    rf = RandomForestClassifier(n_trees)
    rf.fit(X_train[columns_to_use], y_train)
    
    evaluate_configuration(
        rf,
        X_train[columns_to_use],
        y_train,
        X_val[columns_to_use],
        y_val,
        rf_push_train_aucs,
        rf_push_val_aucs,
        rf_push_train_ce,
        rf_push_val_ce,
        rf_push_train_aps,
        rf_push_val_aps
    )
    print(
        f"# Trees: {n_trees} done with train auc: {rf_push_train_aucs[-1]:.4f} and val auc: {rf_push_val_aucs[-1]:.4f}, "
        f"AP train: {rf_push_train_aps[-1]:.4f} and val: {rf_push_val_aps[-1]:.4f}, "
        f"Cross entropy train: {rf_push_train_ce[-1]:.4f} and val: {rf_push_val_ce[-1]:.4f}"
    )
```

    # Trees: 5 done with train auc: 0.7837 and val auc: 0.7439, AP train: 0.0956 and val: 0.0954, Cross entropy train: 0.0687 and val: 0.0921
    # Trees: 25 done with train auc: 0.7846 and val auc: 0.7502, AP train: 0.0986 and val: 0.0970, Cross entropy train: 0.0684 and val: 0.0852
    # Trees: 50 done with train auc: 0.7847 and val auc: 0.7508, AP train: 0.0993 and val: 0.0969, Cross entropy train: 0.0684 and val: 0.0840
    # Trees: 100 done with train auc: 0.7848 and val auc: 0.7529, AP train: 0.0995 and val: 0.0977, Cross entropy train: 0.0684 and val: 0.0817


#### Gradient boosting trees


```python



n_trees_grid = [5, 25, 50, 100]

gbt_push_train_aucs = []
gbt_push_val_aucs = []

gbt_push_train_ce = []
gbt_push_val_ce = []

gbt_push_train_aps = []
gbt_push_val_aps = []

for lr in [0.05, 0.1]:
    for depth in [1, 3, 5]:
        for n_trees in n_trees_grid:
            gbt = GradientBoostingClassifier(
                n_estimators=n_trees, max_depth=depth, learning_rate=lr
            )
            gbt.fit(X_train[columns], y_train)
            
            evaluate_configuration(
                gbt,
                X_train[columns],
                y_train,
                X_val[columns],
                y_val,
                gbt_push_train_aucs,
                gbt_push_val_aucs,
                gbt_push_train_ce,
                gbt_push_val_ce,
                gbt_push_train_aps,
                gbt_push_val_aps
            )
            print(
                f"LR: {lr}, Depth: {depth}, # Trees: {n_trees} done with"
                f"train auc: {gbt_push_train_aucs[-1]:.4f} and val auc: {gbt_push_val_aucs[-1]:.4f}, "
                f"AP train: {gbt_push_train_aps[-1]:.4f} and val: {gbt_push_val_aps[-1]:.4f}, "
                f"Cross entropy train: {gbt_push_train_ce[-1]:.4f} and val: {gbt_push_val_ce[-1]:.4f}"
            )
```

    LR: 0.05, Depth: 1, # Trees: 5 done withtrain auc: 0.6461 and val auc: 0.6515, AP train: 0.0909 and val: 0.0747, Cross entropy train: 0.0729 and val: 0.0672
    LR: 0.05, Depth: 1, # Trees: 25 done withtrain auc: 0.7680 and val auc: 0.7521, AP train: 0.1332 and val: 0.1176, Cross entropy train: 0.0691 and val: 0.0637
    LR: 0.05, Depth: 1, # Trees: 50 done withtrain auc: 0.8163 and val auc: 0.8095, AP train: 0.1554 and val: 0.1377, Cross entropy train: 0.0662 and val: 0.0612
    LR: 0.05, Depth: 1, # Trees: 100 done withtrain auc: 0.8300 and val auc: 0.8292, AP train: 0.1680 and val: 0.1577, Cross entropy train: 0.0644 and val: 0.0589
    LR: 0.05, Depth: 3, # Trees: 5 done withtrain auc: 0.7793 and val auc: 0.7650, AP train: 0.1516 and val: 0.1326, Cross entropy train: 0.0701 and val: 0.0649
    LR: 0.05, Depth: 3, # Trees: 25 done withtrain auc: 0.8221 and val auc: 0.8216, AP train: 0.1810 and val: 0.1682, Cross entropy train: 0.0650 and val: 0.0595
    LR: 0.05, Depth: 3, # Trees: 50 done withtrain auc: 0.8359 and val auc: 0.8386, AP train: 0.1902 and val: 0.1824, Cross entropy train: 0.0631 and val: 0.0573
    LR: 0.05, Depth: 3, # Trees: 100 done withtrain auc: 0.8395 and val auc: 0.8418, AP train: 0.1982 and val: 0.1873, Cross entropy train: 0.0621 and val: 0.0565
    LR: 0.05, Depth: 5, # Trees: 5 done withtrain auc: 0.8336 and val auc: 0.8351, AP train: 0.1798 and val: 0.1645, Cross entropy train: 0.0692 and val: 0.0635
    LR: 0.05, Depth: 5, # Trees: 25 done withtrain auc: 0.8380 and val auc: 0.8401, AP train: 0.2001 and val: 0.1813, Cross entropy train: 0.0634 and val: 0.0579
    LR: 0.05, Depth: 5, # Trees: 50 done withtrain auc: 0.8402 and val auc: 0.8422, AP train: 0.2099 and val: 0.1886, Cross entropy train: 0.0619 and val: 0.0565
    LR: 0.05, Depth: 5, # Trees: 100 done withtrain auc: 0.8437 and val auc: 0.8444, AP train: 0.2205 and val: 0.1876, Cross entropy train: 0.0611 and val: 0.0563
    LR: 0.1, Depth: 1, # Trees: 5 done withtrain auc: 0.6461 and val auc: 0.6515, AP train: 0.0909 and val: 0.0747, Cross entropy train: 0.0709 and val: 0.0654
    LR: 0.1, Depth: 1, # Trees: 25 done withtrain auc: 0.8203 and val auc: 0.8174, AP train: 0.1583 and val: 0.1468, Cross entropy train: 0.0657 and val: 0.0604
    LR: 0.1, Depth: 1, # Trees: 50 done withtrain auc: 0.8321 and val auc: 0.8329, AP train: 0.1674 and val: 0.1574, Cross entropy train: 0.0642 and val: 0.0586
    LR: 0.1, Depth: 1, # Trees: 100 done withtrain auc: 0.8357 and val auc: 0.8372, AP train: 0.1758 and val: 0.1643, Cross entropy train: 0.0633 and val: 0.0578
    LR: 0.1, Depth: 3, # Trees: 5 done withtrain auc: 0.8293 and val auc: 0.8296, AP train: 0.1608 and val: 0.1498, Cross entropy train: 0.0668 and val: 0.0615
    LR: 0.1, Depth: 3, # Trees: 25 done withtrain auc: 0.8366 and val auc: 0.8394, AP train: 0.1910 and val: 0.1799, Cross entropy train: 0.0628 and val: 0.0571
    LR: 0.1, Depth: 3, # Trees: 50 done withtrain auc: 0.8397 and val auc: 0.8418, AP train: 0.1979 and val: 0.1840, Cross entropy train: 0.0621 and val: 0.0565
    LR: 0.1, Depth: 3, # Trees: 100 done withtrain auc: 0.8427 and val auc: 0.8439, AP train: 0.2037 and val: 0.1848, Cross entropy train: 0.0616 and val: 0.0564
    LR: 0.1, Depth: 5, # Trees: 5 done withtrain auc: 0.8377 and val auc: 0.8398, AP train: 0.1908 and val: 0.1715, Cross entropy train: 0.0658 and val: 0.0605
    LR: 0.1, Depth: 5, # Trees: 25 done withtrain auc: 0.8423 and val auc: 0.8431, AP train: 0.2095 and val: 0.1858, Cross entropy train: 0.0617 and val: 0.0566
    LR: 0.1, Depth: 5, # Trees: 50 done withtrain auc: 0.8450 and val auc: 0.8444, AP train: 0.2200 and val: 0.1774, Cross entropy train: 0.0610 and val: 0.0574
    LR: 0.1, Depth: 5, # Trees: 100 done withtrain auc: 0.8471 and val auc: 0.8440, AP train: 0.2327 and val: 0.1751, Cross entropy train: 0.0604 and val: 0.0576



```python
lr_best = 0.05
max_depth_best = 5
n_estimators_best = 50
```


```python
gbt = GradientBoostingClassifier(
    learning_rate=lr_best, max_depth=max_depth_best, n_estimators=n_estimators_best
)
gbt.fit(X_train[columns], y_train)
gbt_fi, fig, ax = plot_feature_importance(gbt, columns)
```


    
![png](hasier_files/hasier_63_0.png)
    


Se puede observar que algunas features no tienen ninguna importancia en el modelo. Por lo tanto, se puede eliminar estas features y volver a entrenar el modelo. Asi el modelo sera mas simple y rapido.


```python
gbt_columns = (
    gbt_fi.loc[gbt_fi["importance"] > 0]
    .sort_values(by="importance", ascending=False)["features"]
    .tolist()
)

gbt_columns = gbt_columns[:10] # Seleccionar las 15 características más importantes
```


```python
gbt_push_train_aucs = []
gbt_push_val_aucs = []

gbt_push_train_ce = []
gbt_push_val_ce = []

gbt_push_train_aps = []
gbt_push_val_aps = []

for i in range (1, len(gbt_columns) + 1):
    gbt = GradientBoostingClassifier(
        learning_rate=lr_best, max_depth=max_depth_best, n_estimators=n_estimators_best
    )
    gbt.fit(X_train[gbt_columns[:i]], y_train)
    
    evaluate_configuration(
        gbt,
        X_train[gbt_columns[:i]],
        y_train,
        X_val[gbt_columns[:i]],
        y_val,
        gbt_push_train_aucs,
        gbt_push_val_aucs,
        gbt_push_train_ce,
        gbt_push_val_ce,
        gbt_push_train_aps,
        gbt_push_val_aps
    )
    print(
        f"Lr: {lr_best}, Depth: {max_depth_best}, # Trees: {n_estimators_best} done with"
        f"train auc: {gbt_push_train_aucs[-1]:.4f} and val auc: {gbt_push_val_aucs[-1]:.4f}, "
        f"AP train: {gbt_push_train_aps[-1]:.4f} and val: {gbt_push_val_aps[-1]:.4f}, "
        f"Cross entropy train: {gbt_push_train_ce[-1]:.4f} and val: {gbt_push_val_ce[-1]:.4f}"
    )
```

    Lr: 0.05, Depth: 5, # Trees: 50 done withtrain auc: 0.6278 and val auc: 0.6458, AP train: 0.0546 and val: 0.0651, Cross entropy train: 0.0714 and val: 0.0640
    Lr: 0.05, Depth: 5, # Trees: 50 done withtrain auc: 0.6461 and val auc: 0.6515, AP train: 0.0954 and val: 0.0771, Cross entropy train: 0.0691 and val: 0.0633
    Lr: 0.05, Depth: 5, # Trees: 50 done withtrain auc: 0.8352 and val auc: 0.8330, AP train: 0.1810 and val: 0.1569, Cross entropy train: 0.0627 and val: 0.0579
    Lr: 0.05, Depth: 5, # Trees: 50 done withtrain auc: 0.8361 and val auc: 0.8339, AP train: 0.1906 and val: 0.1694, Cross entropy train: 0.0626 and val: 0.0576
    Lr: 0.05, Depth: 5, # Trees: 50 done withtrain auc: 0.8376 and val auc: 0.8402, AP train: 0.1950 and val: 0.1814, Cross entropy train: 0.0624 and val: 0.0567
    Lr: 0.05, Depth: 5, # Trees: 50 done withtrain auc: 0.8392 and val auc: 0.8419, AP train: 0.1997 and val: 0.1832, Cross entropy train: 0.0622 and val: 0.0567
    Lr: 0.05, Depth: 5, # Trees: 50 done withtrain auc: 0.8395 and val auc: 0.8423, AP train: 0.2046 and val: 0.1879, Cross entropy train: 0.0621 and val: 0.0566
    Lr: 0.05, Depth: 5, # Trees: 50 done withtrain auc: 0.8395 and val auc: 0.8421, AP train: 0.2064 and val: 0.1872, Cross entropy train: 0.0621 and val: 0.0566
    Lr: 0.05, Depth: 5, # Trees: 50 done withtrain auc: 0.8402 and val auc: 0.8426, AP train: 0.2078 and val: 0.1883, Cross entropy train: 0.0620 and val: 0.0566
    Lr: 0.05, Depth: 5, # Trees: 50 done withtrain auc: 0.8402 and val auc: 0.8426, AP train: 0.2080 and val: 0.1889, Cross entropy train: 0.0620 and val: 0.0565


##### Comparing our models


```python
def get_sumary_metrics( y_true, y_pred):
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
    
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    return precision, recall, pr_thresholds, fpr, tpr, roc_thresholds, auc, ap
```


```python
rf = RandomForestClassifier(100)
rf.fit(X_train[filtered_columns], y_train)
rf_predictions = rf.predict_proba(X_val[filtered_columns])[:, 1]
print("RF all columns trained") 

rf_all = RandomForestClassifier(100)
rf_all.fit(X_train[columns], y_train)
rf_all_predictions = rf_all.predict_proba(X_val[columns])[:, 1]
print("RF filtered columns trained")

gbt = GradientBoostingClassifier(
    learning_rate=lr_best, max_depth=max_depth_best, n_estimators=n_estimators_best
)
gbt.fit(X_train[gbt_columns], y_train)
gbt_predictions = gbt.predict_proba(X_val[gbt_columns])[:, 1]
print("GBT trained")

lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty='l1', C=1e-4, solver='saga')
)

lr.fit(X_train[l1_features], y_train)
lr_predictions = lr.predict_proba(X_val[l1_features])[:, 1]
print("LR trained")
```

    RF all columns trained
    RF filtered columns trained
    GBT trained
    LR trained



```python
(
    rf_push_precision,
    rf_push_recall,
    rf_push_pr_thresholds,
    rf_push_fpr,
    rf_push_tpr,
    rf_push_roc_thresholds,
    rf_push_auc,
    rf_push_ap,
) = get_sumary_metrics(y_val, rf_predictions)

(
    rf_all_push_precision,
    rf_all_push_recall,
    rf_all_push_pr_thresholds,
    rf_all_push_fpr,
    rf_all_push_tpr,
    rf_all_push_roc_thresholds,
    rf_all_push_auc,
    rf_all_push_ap,
) = get_sumary_metrics(y_val, rf_all_predictions)

(
    gbt_push_precision,
    gbt_push_recall,
    gbt_push_pr_thresholds,
    gbt_push_fpr,
    gbt_push_tpr,
    gbt_push_roc_thresholds,
    gbt_push_auc,
    gbt_push_ap,
) = get_sumary_metrics(y_val, gbt_predictions)

(
    lr_push_precision,
    lr_push_recall,
    lr_push_pr_thresholds,
    lr_push_fpr,
    lr_push_tpr,
    lr_push_roc_thresholds,
    lr_push_auc,
    lr_push_ap,
) = get_sumary_metrics(y_val, lr_predictions)
```


```python
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].plot(lr_push_recall, lr_push_precision, label=f"LR AP: {lr_push_ap:.4f}")
ax[0].plot(gbt_push_recall, gbt_push_precision, label=f"GBT AP: {gbt_push_ap:.4f}")
ax[0].plot(rf_push_recall, rf_push_precision, label=f"RF AP: {rf_push_ap:.4f}")
ax[0].plot(
    rf_all_push_recall, rf_all_push_precision, label=f"RF all AP: {rf_all_push_ap:.4f}"
)

ax[0].set_xlabel("recall")
ax[0].set_ylabel("precision")

ax[1].plot(lr_push_fpr, lr_push_tpr, label=f"LR AUC: {lr_push_auc:.4f}")
ax[1].plot(gbt_push_fpr, gbt_push_tpr, label=f"GBT AUC: {gbt_push_auc:.4f}")
ax[1].plot(rf_push_fpr, rf_push_tpr, label=f"RF AUC: {rf_push_auc:.4f}")
ax[1].plot(
    rf_all_push_fpr, rf_all_push_tpr, label=f"RF all AUC: {rf_all_push_auc:.4f}"
)
ax[1].set_xlabel("FPR")
ax[1].set_ylabel("TPR")

ax[0].legend()
ax[1].legend()
```




    <matplotlib.legend.Legend at 0x7f985ac96950>




    
![png](hasier_files/hasier_71_1.png)
    


Se puede observa que el modelo de GBT es el mejor.
