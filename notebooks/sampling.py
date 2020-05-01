from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler


def do_sampling(upsampling, train_features, train_truth, smote=False):
  X_train = train_features
  y_train = train_truth.label


  if smote is False:
    X = pd.concat([X_train, y_train], axis=1)
    
    completed = X[X.label==0]
    drop_out = X[X.label==1]

    if upsampling is True:
      sampled = resample(completed,
                            replace=True, # sample with replacement
                            n_samples=len(drop_out), # match number in majority class
                            random_state=27) # reproducible results
      sampled = pd.concat([drop_out, sampled])
    elif upsampling is False:
      sampled = resample(drop_out,
                            replace=False, # sample with replacement
                            n_samples=len(completed), # match number in majority class
                            random_state=27) # reproducible results
      sampled = pd.concat([sampled, completed])
    
    if upsampling is not None:
      sampled = sampled.sample(frac=1).reset_index(drop=True)
      print(sampled.label.value_counts())
      y_train = sampled.label
      X_train = sampled.drop('label', axis=1)
  
  elif smote is True:

    feature_columns = X_train.columns

    if upsampling is True:
      
      upsample = SMOTE()
      X_train, y_train = upsample.fit_resample(X_train, y_train)
    
    elif upsampling is False:
      
      over = SMOTE(sampling_strategy=0.4)
      under = RandomUnderSampler(sampling_strategy=0.5)
      
      steps = [('o', over), ('u', under)]
      pipeline = Pipeline(steps=steps)

      X_train, y_train = pipeline.fit_resample(X_train, y_train)


    features = pd.DataFrame(data=X_train, columns=columns)
    label = pd.DataFrame(data=y_train, columns=['label'])
    
    X = pd.concat([features, label], axis=1)
    
    X = X.sample(frac=1).reset_index(drop=True)
    
    print(X.label.value_counts())
    
    y_train = X.label
    X_train = X.drop('label', axis=1)

  return X_train, y_train




#=================================================



# setting up testing and training sets
# smote - False : upsample = True, upsample = False, upsample = None
# To try - upsample = True and upsample = None(gives original imbalanced class)

# smote - True : upsample = True, upsample = False(upsamples minority and downsamples majority)
# To try - upsample = True and upsample = False(upsamples minority and downsamples majority)


smote = True
upsample = None
X_train, y_train = do_sampling(upsample, train_features, train_truth, smote)
print(X_train.shape)


X_test = test_features
y_test = test_truth.label
