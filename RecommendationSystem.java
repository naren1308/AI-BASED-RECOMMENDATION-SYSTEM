import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RecommendationSystem {
    private static final Logger LOGGER = Logger.getLogger(RecommendationSystem.class.getName());
    private static final String DATA_FILE = "data.csv";
    private static final int NEIGHBORHOOD_SIZE = 2; 
    private static final int USER_ID = 1;  
    private static final int NUM_RECOMMENDATIONS = 3; 

    public static void main(String[] args) {
        try {
           
            DataModel model = loadDataModel(DATA_FILE);

          
            Recommender recommender = createRecommender(model);

           
            generateRecommendations(recommender, USER_ID, NUM_RECOMMENDATIONS);

           
            evaluateRecommender(recommender, model);
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error occurred in recommendation system: ", e);
        }
    }

    private static DataModel loadDataModel(String filePath) throws Exception {
        File file = new File(filePath);
        if (!file.exists()) {
            throw new Exception("Data file not found: " + filePath);
        }
        return new FileDataModel(file);
    }

    private static Recommender createRecommender(DataModel model) throws Exception {
        UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
        UserNeighborhood neighborhood = new NearestNUserNeighborhood(NEIGHBORHOOD_SIZE, similarity, model);
        return new GenericUserBasedRecommender(model, neighborhood, similarity);
    }

    private static void generateRecommendations(Recommender recommender, int userId, int numRecommendations) {
        try {
            List<RecommendedItem> recommendations = recommender.recommend(userId, numRecommendations);
            if (recommendations.isEmpty()) {
                System.out.println("No recommendations found for user ID " + userId);
            } else {
                System.out.println("Recommendations for User ID " + userId + ":");
                for (RecommendedItem recommendation : recommendations) {
                    System.out.println("Item: " + recommendation.getItemID() + " | Score: " + recommendation.getValue());
                }
            }
        } catch (Exception e) {
            LOGGER.log(Level.WARNING, "Error generating recommendations: ", e);
        }
    }

    private static void evaluateRecommender(Recommender recommender, DataModel model) {
        try {
            RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
            double score = evaluator.evaluate(recommender, null, model, 0.9, 1.0);
            System.out.println("Recommender Evaluation Score: " + score);
        } catch (Exception e) {
            LOGGER.log(Level.WARNING, "Error evaluating recommender: ", e);
        }
    }
}