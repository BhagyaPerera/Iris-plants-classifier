
%part 1-training data
%read normalized data from dataset excel sheet for training
clc;
data1=xlsread('dataset.xlsx','Training','B2:J76');
P1=data1(1:end,2);
P2=data1(1:end,4);
P3=data1(1:end,6);
P4=data1(1:end,8);
t=data1(1:end,9);
P=[P1,P2,P3,P4];

%initialize weight functions
W1=ones(3,4);
W2=rand(1,3);


%epochs can be 500,1000,5000

epochs=500;
%initialize variables
niu=0.7;
MSE=ones(epochs,1);
error=ones(75,1);
err=ones(75,1);
output=ones(75,1);
   


for m=1:epochs
    
    for k=1:75
        
        input=P(k,1:4)';
        n1=W1*input;
        o1=logsig(n1);
        %output of the input layer concerned as input to the hidden layer
        
        n2=W2*o1;
        output(k,1)=logsig(n2);
        
        
        error(k,1)=t(k,1)-output(k,1);
        
        
        err(k,1)=error(k,1)^2;
       
       
        S2=output(k,1)*(1-output(k,1))'*error(k,1);
        
       a=[1;1;1];
       j=a-o1;
        S1=o1'*j*S2*W2;
        
        
        dweight1=input*S1*niu;
        dweight2=niu*S2*o1';
        
        W1=W1+dweight1';
        W2=W2+dweight2;
     
    end
    MSE(m,1)=mean(err);
    
    
     
end

%plot MSE against epochs
figure(1);
line(1:epochs,MSE);
title('Epochs vs MSE')
xlabel('No of Epochs')
ylabel('MSE')


% end of training




%part 2- test data

%read data from dataset excel sheet for testing
data2=xlsread('dataset.xlsx','Test','A2:I76');
T1=data2(1:end,2);
T2=data2(1:end,4);
T3=data2(1:end,6);
T4=data2(1:end,8);
y=data2(1:end,9);
T=[T1,T2,T3,T4];

%initialize variables
classifiedSetosa=0;
classifiedVersicolor=0;
classifiedVirginnica=0;
OT=ones(75,1);
e=ones(75,1);

 %test Iris Setosa classification performance
   for a=1:25
        
        input=T(a,1:4)';
        n1=W1*input;
        o1=logsig(n1);
        %output of the input layer concerned as input to the hidden layer
        
        n2=W2*o1;
        OT(a,1)=logsig(n2);
        
        
        error(a,1)=y(a,1)-OT(a,1);
        e(a,1)=abs(error(a,1));
        if e(a,1)<0.01
            classifiedSetosa=classifiedSetosa+1;
        end
       
   end
    
   %test Iris Versicolor classification performance
    
    for a=26:50
        
        input=T(a,1:4)';
        n1=W1*input;
        o1=logsig(n1);
        %output of the input layer concerned as input to the hidden layer
        
        n2=W2*o1;
        OT(a,1)=logsig(n2);
        
        
        error(a,1)=y(a,1)-OT(a,1);
        e(a,1)=abs(error(a,1));
        if e(a,1)<0.01
            classifiedVersicolor=classifiedVersicolor+1;
        end    
   
        
    end
    
   %Test Iris virginnnica classification performance  
   for a=51:75
        
        input=T(a,1:4)';
        n1=W1*input;
        o1=logsig(n1);
        %output of the input layer concerned as input to the hidden layer
        
        n2=W2*o1;
        OT(a,1)=logsig(n2);
        
        
        error(a,1)=y(a,1)-OT(a,1);
        e(a,1)=abs(error(a,1));
        if e(a,1)<0.01
         classifiedVirginnica=classifiedVirginnica+1;
        
        end    
   
   end   
 
 % scatter plot of classification results
 figure(2);
 scatter(OT(1:25,1),1:25);
 hold on;
 scatter(OT(26:50,1),26:50);
 scatter(OT(51:75,1),51:75);
 hold off;
 
        
        


 
%print the classification result
fprintf('ClassifiedSetosa \t ClassifiedVersicolor \t Classified Virginnica\n'); 
fprintf('%d\t,%d\t,%d\t',[classifiedSetosa,classifiedVersicolor,classifiedVirginnica]);

 
 
 
        

 


        
            
