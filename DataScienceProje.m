clearvars;  % Tüm değişkenleri temizler
clc;        % Komut penceresini temizler

% Read Excel file
df = readtable('/home/hp-laserjet-wrt8070/Desktop/MI.xlsm');

% Sutun1 adlı sütunu kaldır(sınıflandırmayı etkiliyor)
df = removevars(df, 'Sutun1');

% class_labels'ı oluşturun
class_labels = df.Sutun2;

df = removevars(df, 'Sutun2');

% NaN değerleri median  doldurma
df = fillmissing(df, 'constant', mean(df{:,:}));

% Min-max normalization
df = (df - min(df)) ./ (max(df) - min(df));

% Convert df to a numeric matrix
df_numeric = table2array(df);


%RFE:(Matlabdeki Karşılığı Relieff)
%RFE, bir veri kümesindeki özelliklerin önem sırasını belirlemek ve gereksiz özellikleri elemek için kullanılan bir özellik seçimi tekniğidir. 
%Adından da anlaşılacağı gibi, bu yöntem özellikleri bir bir elemekte ve her adımda en az etkili olan özellikleri çıkartarak bir model eğitir. 
%En sonunda, belirlenen bir özellik sayısı veya başka bir kriter sağlanana kadar bu işlem devam eder.
%RFE genellikle destek vektör makineleri (SVM) gibi modellerle birlikte kullanılır.

% Apply Relief feature selection
[selected_features, ~] = relieff(df_numeric, class_labels, 10);

% Extract the selected features
df_selected = df_numeric(:, selected_features);

% En iyi 10 özelliği seçin
best_features = df_selected(:, selected_features(1:10));

% Korelasyon matrisini hesaplayın
best_features_matrix = corrcoef(best_features);

% Korelasyon matrisi
figure;
h = heatmap(best_features_matrix);
h.Title = 'en iyi 10 özellik için korelasyon Matrisi ';
h.XLabel = 'Özellik';
h.YLabel = 'Özellik';


%PCA, veri setindeki değişkenlik miktarını azaltmak için kullanılan bir boyut azaltma tekniğidir. 
%Bu, çok boyutlu veri setlerindeki karmaşıklığı azaltmak ve veriyi daha anlaşılır hale getirmek için kullanılır. 
%PCA, veri setindeki değişkenler arasındaki korelasyonu temel alarak yeni bir değişken seti oluşturur, 
%bu değişken seti orijinal verinin varyansını maksimize eder.

% PCA uygula
[coeff, score, latent, tsquared, explained] = pca(best_features_matrix);
% PCA sonuçlarını ekrana yazdır
disp('Katsayılar (Eigenvectors):');
disp(coeff);
disp('Skorlar (Yeni veri uzayı):');
disp(score);
disp('Özdeğerler (Variance explained by each PC):');
disp(latent);
disp('Her bileşenin açıkladığı varyans yüzdesi:');
disp(explained);




%cvpartition, 
%çapraz doğrulama (cross-validation) için veri setini bölmenizi sağlayan MATLAB'de bir fonksiyondur. 
%Çapraz doğrulama, bir makine öğrenimi modelinin gerçek dünya verilerine nasıl genelleşeceğini daha iyi değerlendirmek için kullanılır. 
%Temel fikir, veri setini eğitim ve test alt kümelerine ayırmak ve 
%modelin genelleştirilebilirliğini test etmek için bu alt kümeler üzerinde birden fazla kez modelin eğitilip test edilmesidir.

% Split data into training and test sets using cvpartition
cv = cvpartition(size(best_features, 1), 'Holdout', 0.1);
train_indices = training(cv);
test_indices = test(cv);

% Split data into training and test sets
X_train = best_features(train_indices, :);
X_test = best_features(test_indices, :);
y_train = class_labels(train_indices);
y_test = class_labels(test_indices);



% Modellerin eğitimi ve performans ölçümü

%LDA 
%LDA (Linear Discriminant Analysis), 
%temelde sınıflar arasındaki farklılıkları ölçerek 
%sınıfları birbirinden ayırmak için kullanılan bir öğrenme ve sınıflandırma yöntemidir. Özellikle, 
%özniteliklerin bir veri noktasının hangi sınıfa ait olduğunu belirleme yeteneğini artırmak için kullanılır.

lda_model = fitcdiscr(X_train, y_train, 'discrimType', 'pseudoLinear');
y_pred = predict(lda_model, X_test);
accuracy = sum(y_pred == y_test) / length(y_test);
cm = confusionmat(y_test, y_pred);

fprintf('LDA Model Doğruluk Değeri: %.2f\n', accuracy);
% LDA Model Confusion Matrix
figure;
confusionchart(cm, lda_model.ClassNames);
title('LDA Model Confusion Matrix');

%SVM'nin temel amacı, 
% veri noktalarını sınıflara ayırmak için bir karar sınırı (hiper düzlem) belirlemektir. 
% Bu karar sınırı, veri noktalarının en iyi şekilde sınıflara ayrılmasını sağlamak için belirlenir. 
% SVM, bu amaçla, sınıflar arasındaki en büyük marjı (mesafe) maksimize etmeye çalışır. 
% Bu marj, karar sınırı ile en yakın veri noktaları arasındaki mesafedir ve bu noktalara "destek vektörleri" denir.
svm_model = fitcecoc(best_features, class_labels, 'Learners', 'svm', 'FitPosterior', true);
y_pred = predict(svm_model, X_test);
accuracy = sum(y_pred == y_test) / length(y_test);
cm = confusionmat(y_test, y_pred);

fprintf('SVM Model Doğruluk Değeri: %.2f\n', accuracy);
% SVM Model Confusion Matrix
figure;
confusionchart(cm, svm_model.ClassNames);
title('SVM Model Confusion Matrix');

%DT (Decision Tree), karar ağacı olarak da bilinen, 
% sınıflandırma ve regresyon problemleri için kullanılan bir makine öğrenimi algoritmasıdır. 
% Temel amacı, bir veri setindeki özelliklerin değerlerine göre kararlar vererek veri noktalarını sınıflandırmak veya 
% bir hedef değişkenin değerini tahmin etmektir.
dt_model = fitctree(best_features, class_labels);
y_pred = predict(dt_model, X_test);
accuracy = sum(y_pred == y_test) / length(y_test);
cm = confusionmat(y_test, y_pred);

fprintf('Decision Tree Model Doğruluk Değeri: %.2f\n', accuracy);
% Decision Tree Model Confusion Matrix
figure;
confusionchart(cm, dt_model.ClassNames);
title('Decision Tree Model Confusion Matrix');


%RF (Random Forest), 
% karar ağaçları (Decision Trees) üzerine kurulu bir makine öğrenimi algoritmasıdır. 
% Temel fikri, birden çok karar ağacının bir araya gelerek güçlü bir sınıflandırıcı veya 
% regresyon modeli oluşturmasıdır. Her bir karar ağacı veri setinin rastgele örneklenmiş alt kümeleriyle eğitilir ve 
% bu alt kümelerin üzerinde rastgele özellikler seçilerek ağaçlar çeşitlendirilir.
rf_model = fitcensemble(best_features, class_labels, 'Method', 'bag', 'NumLearningCycles', 4);
y_pred = predict(rf_model, X_test);
accuracy = sum(y_pred == y_test) / length(y_test);
cm = confusionmat(y_test, y_pred);

fprintf('Random Forest Model Doğruluk Değeri: %.2f\n', accuracy);

% Random Forest Model Confusion Matrix
figure;
confusionchart(cm, rf_model.ClassNames);
title('Random Forest Model Confusion Matrix');