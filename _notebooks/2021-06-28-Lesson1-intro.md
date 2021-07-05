# Lesson 1 - FastAI


## Your First Model


```python
from fastai.vision.all import * #IMPORT 
path = untar_data(URLs.PETS)/'images' #DATA SET 

def is_cat(x): return x[0].isupper() #Labels for the dataset (This dataset cat labels begin w/ uppercase letter)

#Create dataset (Training data, test data) and correctly gets imgs w/ labels
dls = ImageDataLoaders.from_name_func( 
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate) #Creating architecture
learn.fine_tune(1) #Training
```

### Sidebar: This Book Was Written in Jupyter Notebooks


```python
img = PILImage.create(image_cat())
img.to_thumb(192)
```




![png](output_4_0.png)



### End sidebar


```python
uploader = widgets.FileUpload()
uploader
```


```python
img = PILImage.create(uploader.data[0])
img.to_thumb(192)
```




![png](output_7_0.png)




```python
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
```





    Is this a cat?: True.
    Probability it's a cat: 0.999998


### Limitations Inherent To Machine Learning

From this picture we can now see some fundamental things about training a deep learning model:

- A model cannot be created without data.
- A model can only learn to operate on the patterns seen in the input data used to train it.
- This learning approach only creates *predictions*, not recommended *actions*.
- It's not enough to just have examples of input data; we need *labels* for that data too (e.g., pictures of dogs and cats aren't enough to train a model; we need a label for each one, saying which ones are dogs, and which are cats).

Generally speaking, we've seen that most organizations that say they don't have enough data, actually mean they don't have enough *labeled* data. If any organization is interested in doing something in practice with a model, then presumably they have some inputs they plan to run their model against. And presumably they've been doing that some other way for a while (e.g., manually, or with some heuristic program), so they have data from those processes! For instance, a radiology practice will almost certainly have an archive of medical scans (since they need to be able to check how their patients are progressing over time), but those scans may not have structured labels containing a list of diagnoses or interventions (since radiologists generally create free-text natural language reports, not structured data). We'll be discussing labeling approaches a lot in this book, because it's such an important issue in practice.

Since these kinds of machine learning models can only make *predictions* (i.e., attempt to replicate labels), this can result in a significant gap between organizational goals and model capabilities. For instance, in this book you'll learn how to create a *recommendation system* that can predict what products a user might purchase. This is often used in e-commerce, such as to customize products shown on a home page by showing the highest-ranked items. But such a model is generally created by looking at a user and their buying history (*inputs*) and what they went on to buy or look at (*labels*), which means that the model is likely to tell you about products the user already has or already knows about, rather than new products that they are most likely to be interested in hearing about. That's very different to what, say, an expert at your local bookseller might do, where they ask questions to figure out your taste, and then tell you about authors or series that you've never heard of before.

## Deep Learning Is Not Just for Image Classification
### The below is a segmentation model


```python
path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func( #Segmentation
    path, bs=8, fnames = get_image_files(path/"images"),
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
    codes = np.loadtxt(path/'codes.txt', dtype=str)
)

learn = unet_learner(dls, resnet34)
learn.fine_tune(8)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.821857</td>
      <td>3.201599</td>
      <td>00:05</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.189840</td>
      <td>1.802189</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.831589</td>
      <td>1.741264</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.633543</td>
      <td>1.287880</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.459935</td>
      <td>1.189295</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.315327</td>
      <td>1.003694</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.188064</td>
      <td>0.926959</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.080196</td>
      <td>0.872466</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.997371</td>
      <td>0.871319</td>
      <td>00:04</td>
    </tr>
  </tbody>
</table>



```python
learn.show_results(max_n=2, figsize=(10,12))
```






![png](output_12_1.png)


### The below is a natural language model


```python
from fastai.text.all import *

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(2, 1e-2)
```


```python
learn.predict("I really liked that movie!")
```








    ('pos', tensor(1), tensor([0.0228, 0.9772]))



### The below is a salary prediction model


```python
from fastai.tabular.all import *
path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
    cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics=accuracy)
```






```python
learn.fit_one_cycle(3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.372949</td>
      <td>0.361306</td>
      <td>0.833077</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.354939</td>
      <td>0.348455</td>
      <td>0.841830</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.349614</td>
      <td>0.347378</td>
      <td>0.840756</td>
      <td>00:05</td>
    </tr>
  </tbody>
</table>


### The below is a reccomendation model


```python
from fastai.collab import *
path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path/'ratings.csv')
learn = collab_learner(dls, y_range=(0.5,5.5))
learn.fine_tune(10)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.523757</td>
      <td>1.424118</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.376706</td>
      <td>1.363828</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.282471</td>
      <td>1.173324</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.034186</td>
      <td>0.848724</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.805283</td>
      <td>0.694374</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.709625</td>
      <td>0.654900</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.652975</td>
      <td>0.645875</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.634861</td>
      <td>0.639299</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.611313</td>
      <td>0.637229</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.617857</td>
      <td>0.636715</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.612095</td>
      <td>0.636567</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



```python
learn.show_results()
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>rating_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>87.0</td>
      <td>48.0</td>
      <td>5.0</td>
      <td>4.045295</td>
    </tr>
    <tr>
      <th>1</th>
      <td>73.0</td>
      <td>92.0</td>
      <td>4.0</td>
      <td>4.072179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66.0</td>
      <td>26.0</td>
      <td>3.0</td>
      <td>4.015839</td>
    </tr>
    <tr>
      <th>3</th>
      <td>66.0</td>
      <td>30.0</td>
      <td>3.0</td>
      <td>3.367572</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>46.0</td>
      <td>3.5</td>
      <td>3.269851</td>
    </tr>
    <tr>
      <th>5</th>
      <td>82.0</td>
      <td>84.0</td>
      <td>4.0</td>
      <td>3.817361</td>
    </tr>
    <tr>
      <th>6</th>
      <td>90.0</td>
      <td>79.0</td>
      <td>4.0</td>
      <td>4.012848</td>
    </tr>
    <tr>
      <th>7</th>
      <td>61.0</td>
      <td>65.0</td>
      <td>4.0</td>
      <td>3.507185</td>
    </tr>
    <tr>
      <th>8</th>
      <td>88.0</td>
      <td>7.0</td>
      <td>4.5</td>
      <td>4.166433</td>
    </tr>
  </tbody>
</table>


## Questionnaire

It can be hard to know in pages and pages of prose what the key things are that you really need to focus on and remember. So, we've prepared a list of questions and suggested steps to complete at the end of each chapter. All the answers are in the text of the chapter, so if you're not sure about anything here, reread that part of the text and make sure you understand it. Answers to all these questions are also available on the [book's website](https://book.fast.ai). You can also visit [the forums](https://forums.fast.ai) if you get stuck to get help from other folks studying this material.

For more questions, including detailed answers and links to the video timeline, have a look at Radek Osmulski's [aiquizzes](http://aiquizzes.com/howto).

1. **Do you need these for deep learning?**

   - Lots of math T / **F**
   - Lots of data T / **F**
   - Lots of expensive computers T / **F**
   - A PhD T / **F**
   
1. **Name five areas where deep learning is now the best in the world.** </br>
Vision, Natural language processing, Medicine, Robotics, and Games
1. **What was the name of the first device that was based on the principle of the artificial neuron?** </br>
Mark I Perceptron
1. **Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?** </br>
Processing units, State of activation, Output function, Pattern of connectivity, Propagation rule, Activation rule, Learning rule, Environment
1. **What were the two theoretical misunderstandings that held back the field of neural networks?** </br>
Single layer network unable to learn simple mathimatical functions.</br>
More layers make network too big and slow to be useful.
1. **What is a GPU?**</br>
A graphics card is a processor that can handle 1000's of tasks at the same time. Particularly great for deep learning.
1. **Open a notebook and execute a cell containing: `1+1`. What happens?** </br>
2
1. **Follow through each cell of the stripped version of the notebook for this chapter. Before executing each cell, guess what will happen.**
1. **Complete the Jupyter Notebook online appendix.**
1. **Why is it hard to use a traditional computer program to recognize images in a photo?** </br>
They are missing the weight assignment needed to recognize patterns within images to accomplish the task.
1. **What did Samuel mean by "weight assignment"?**</br>
The weight is another form of input that has direct influence on the model's performance.
1. **What term do we normally use in deep learning for what Samuel called "weights"?**</br> 
Parameters
1. **Draw a picture that summarizes Samuel's view of a machine learning model.**</br>

1. **Why is it hard to understand why a deep learning model makes a particular prediction?**</br> 
There are many layers, each with numerous neurons. Therefore, it gets complex really fast what each neuron is looking for when viewing an image, and how that impacts the perediction.  
1. **What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?**</br>
Universal approximation theorem 
1. **What do you need in order to train a model?**</br>
Data with labels
1. **How could a feedback loop impact the rollout of a predictive policing model?**</br> 
The more the model is used the more biased the data becomes, and therefore, the more bias the model becomes. 
1. **Do we always have to use 224×224-pixel images with the cat recognition model?**</br>
No.
1. **What is the difference between classification and regression?**</br>
Classification is about categorizing/labeling objects. </br>
Regression is about predicting numerical quantities, such as temp.
1. **What is a validation set? What is a test set? Why do we need them?**</br>
The validation set measures the accuracy of the model during training. </br>
The test set is used during the final evaluation to test the accuracy of the model. </br></br>
We need both of them because the validation set could cause some bias in the model as we would are fitting the model towards it during training. However, the test set removes this and evaluates the model on unseen data, thereby, giving an accurate metric of accuracy.  
1. **What will fastai do if you don't provide a validation set?**</br>
Fastai will automatically create a validation dataset for us. 
1. **Can we always use a random sample for a validation set? Why or why not?**</br>
It is not reccomended where order is neccessary, example ordered by time.
1. **What is overfitting? Provide an example.**</br>
This is when the model begins to fit to the training data rather than generalizing for similar unseen datasets. For example a model that does amazing on the training data, but performs poorly on test data: Good indication that model may have overfitted. 
1. **What is a metric? How does it differ from "loss"?**</br>
The loss is the value calculated by the model to determine the impact each neuron has on the end result: Therefore, the value is used by models to measure its performance. The metric gives us, humans, an overall value of how accurate the model was: Therefore, a value we use to understand the models performance. 
1. **How can pretrained models help?**</br>
A pretrained model already has the fundementals. Therefore, it can use this prior knowledge to learn faster and perform better on similer datasets.
1. **What is the "head" of a model?**</br> 
The final layers from the pretrained model that have been replaced with new layers (w/ randomized weights) to better align with our dataset. These final layers are often the only thing trained while the rest of the model is frozen.
1. **What kinds of features do the early layers of a CNN find? How about the later layers?**</br>
The early layers often extract simple features like edges. </br>
The later layers are more complex and can identify advanced features like faces.
1. **Are image models only useful for photos?**</br>
No. Lots of other forms of data can be converted into images that can be used to solve such non-photo data problems. 
1. **What is an "architecture"?**</br>
This is the structure of the model we use to solve the problem.
1. **What is segmentation?**</br>
Method of labeling all pixels within an image and masking it.
1. **What is `y_range` used for? When do we need it?**</br>
Specifies the range of values that can be perdicted by model. For example, movie rating's 0-5.
1. **What are "hyperparameters"?**</br>
These are the parameters that we can adjust to help the model perform better (Ex: Epochs). 
1. **What's the best way to avoid failures when using AI in an organization?**</br>
Begin with the most simplest model and then slowly building up to more complexity. This way you have something working and don't get lost as you add onto the model. 


### Further Research

Each chapter also has a "Further Research" section that poses questions that aren't fully answered in the text, or gives more advanced assignments. Answers to these questions aren't on the book's website; you'll need to do your own research!

1. **Why is a GPU useful for deep learning? How is a CPU different, and why is it less effective for deep learning?** </br>
Modern GPUs provide a far superior processing power, memory bandwidth, and efficiency over the CPU. 

1. **Try to think of three areas where feedback loops might impact the use of machine learning. See if you can find documented examples of that happening in practice.** </br>
I believe feedback loops are primarly great for recommendation models. This is because the feedback loops create a bias model. For example, if a viewer like a movie, he/she will like similer movies. Being bias here towards particular types of movie is the best way to keep the viewer engaged.


```python

```
