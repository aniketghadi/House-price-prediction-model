
#First we have to import libraries
#Think of libraries as "pre-written programs" that help us accelerate what we do in Python

#Gradio is a web interface library for deploying machine learning models
import gradio as gr

#Pickle is a library that lets us work with machine learning models, which in Python are typically in a "pickle" file format
import pickle

#Orange is the Python library used by... well, Orange!
from Orange.data import *

#This is called a function. This function can be "called" by our website (when we click submit). Every time it's called, the function runs.
#Within our function, there are inputs (bedrooms1, bathrooms1, etc.). These are passed from our website front end, which we will create further below.
def make_prediction(bedrooms1, bathrooms1, stories1, mainroad1,guestroom1,basement1,hotwaterheating1,airconditioning1,parking1,prefarea1,furnishingstatus1):
    
	#Because we already trained a model on these variables, any inputs we feed to our model has to match the inputs it was trained on.
	#Even if you're not familiar with programming, you can probably decipher the below code.
    bedrooms=DiscreteVariable("bedrooms",values=["1","2","3","4","5","6"])
    bathrooms=DiscreteVariable("bathrooms",values=["1","2","3"])
    stories=DiscreteVariable("stories",values=["1","2","3","4"])
    mainroad=DiscreteVariable("mainroad",values=["yes","no"])
    guestroom=DiscreteVariable("guestroom",values=["yes","no"])
    basement=DiscreteVariable("basement",values=["yes","no"])
    hotwaterheating=DiscreteVariable("hotwaterheating",values=["yes","no"])
    airconditioning=DiscreteVariable("airconditioning",values=["yes","no"])
    parking=DiscreteVariable("parking",values=["0","1","2","3"])
    prefarea=DiscreteVariable("prefarea",values=["yes","no"])
    furnishingstatus=DiscreteVariable("furnishingstatus",values=['furnished','semi-furnished','unfurnished'])
    
	#This code is a bit of housekeeping. 
	#Since our model is expecting discrete inputs (just like in Orange), we need to convert our numeric values to strings
    bedrooms1=str(bedrooms1)
    bathrooms1=str(bathrooms1)
    stories1=str(stories1)
    parking1=str(parking1)
	
	#A domain is essentially an Orange file definition. Just like the one you set with the "file node" in the tool.
    domain=Domain([bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus])
	
	#Our data is the data being passed by the website inputs. This gets mapped to our domain, which we defined above.
    data=Table(domain,[[bedrooms1, bathrooms1, stories1, mainroad1,guestroom1,basement1,hotwaterheating1,airconditioning1,parking1,prefarea1,furnishingstatus1]])
	
	#Next, we can work on our predictions!

	#This tiny piece of code loads our model (pickle load).
    with open("model_old.pkcls", "rb") as f:
		#Then feeds our data into the model, then sets the "preds" variable to the prediction output for our class variable, which is price.	
        clf  = pickle.load(f)
        ar=clf(data)
        preds=clf.domain.class_var.str_val(ar)
        preds="$"+preds
		#Finally, we send the prediction to the website.
        return preds

#Now that we have defined our prediction function, we need to create our web interface.
#This code creates the input components for our website. Gradio has this well documented and it's pretty easy to modify.
NumberOfBedrooms=gr.Slider(minimum=1,maximum=6,step=1,label="How many bedrooms?")
NumberOfBathrooms=gr.Slider(minimum=1,maximum=3,step=1,label="How many bathrooms?")
NumberOfStories=gr.Slider(minimum=1,maximum=4,step=1,label="How many stories?")
OnMainRoad=gr.Dropdown(["yes","no"],label="Is the house on a main road?")
HasGuestRoom=gr.Dropdown(["yes","no"],label="Does the house have a guest room?")
HasBasement=gr.Dropdown(["yes","no"],label="Does the house have a basement?")
HasHotWaterHeating=gr.Dropdown(["yes","no"],label="Does the house have hot water heating?")
HasAC=gr.Dropdown(["yes","no"],label="Does the house have air conditioning?")
parkingstatus=gr.Slider(minimum=0,maximum=3,step=1,label="How many parking spots?")
PreferredArea=gr.Dropdown(["yes","no"],label="Does the house have Preferred Area ?")
Furnished=gr.Dropdown(['furnished','semi-furnished','unfurnished'],label='Is the house furnished?')

# Next, we have to tell Gradio what our model is going to output. In this case, it's going to be a text result (house prices).
output = gr.Textbox(label="House price estimate:")

#Then, we just feed all of this into Gradio and launch the web server. 
#Our fn (function) is our make_prediction function above, which returns our prediction based on the inputs.
app = gr.Interface(fn = make_prediction, inputs=[NumberOfBedrooms, NumberOfBathrooms, NumberOfStories,OnMainRoad,HasGuestRoom,HasBasement,HasHotWaterHeating,HasAC,parkingstatus,PreferredArea, Furnished], outputs=output)
app.launch()
