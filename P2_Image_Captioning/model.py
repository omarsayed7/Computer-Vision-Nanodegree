import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
                 
        self.hidden_size= hidden_size

        '''
        caption embedding to fixed size 
        '''
        self.caption_embedding = nn.Embedding(vocab_size,
                                              embed_size)
        '''
        LSTM  layer defiend with embedded dimention, 
        hidden state size, number of layers 
        '''
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            num_layers,
                            batch_first= True)
        '''
        Output of the lstm goes through fully connected layer, 
        to give logits with the same vocab size length 
        '''
        self.hidden2vocab = nn.Linear(hidden_size,vocab_size)
      
 
    
    def forward(self, features, captions):
        '''
        args:-
        features: embedded feature vector of the image (coming from the encoder network)
        captions: tokenized captions 
        '''
        #discatd <end> word to avoid predicting it.
        captions = captions[:,:-1]
        
        #we can get the batch size from features dimention that help us init the h,c
        batch_size = features.shape[0]
        self.hidden= self.init_hidden(batch_size)
        
        #print(captions.shape)
        
        #embedding layer applied to captions 
        embeds = self.caption_embedding(captions)
        #print(embeds.shape)
        
        #concat the image embedding vector with captions embedding vector [ref]: knoweledge center question
        full_embeds = torch.cat((features.unsqueeze(1), embeds), dim = 1)
        
        out, self.hidden = self.lstm(full_embeds, self.hidden)
        
        out = self.hidden2vocab(out)
        return out
    
    def init_hidden(self, batch_size):
        '''
        the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        '''
        
        hidden, cell=  (torch.randn((1, batch_size, self.hidden_size)),
                torch.randn((1, batch_size, self.hidden_size)))
        
        hidden = hidden.cuda()
        cell = cell.cuda()
        
        return hidden, cell
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #Get the batch size to initialize the hidden state for RNN-Decoder
        batchsize= inputs.shape[0]
        hidden= self.init_hidden(batchsize)
        #Output list that contains the words sampled from the Decoder one at a time
        output_sentense = []
        #ensure we are in validation 
        with torch.no_grad():
            #loop over the maximum length of the sentense 
            for i in range(max_len):
                #give first the input feature vector 
                out_lstm, hidden = self.lstm(inputs, hidden)
                outputs = self.hidden2vocab(out_lstm)
                #outputs shape is (1,1,vocab_size)
                outputs = outputs.squeeze(1)
                #outputs shape is (1, vocab_size)
                outputs = outputs.argmax(dim=1)
                #append the output word in the sentense list
                output_sentense.append(outputs.cpu().item())
                
                #if the predicted word is <end> break the loop
                if outputs == 1: 
                    #that is the end word 
                    break
                
                #get the last word then embedded it and feed it back to the Decoder-RNN
                inputs = self.caption_embedding(outputs)
                inputs = inputs.unsqueeze(0) #input shape (1, 1, embed_size)
                
       
        return output_sentense        
        
        
        