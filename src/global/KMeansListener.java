package global;

/**
 * Defines object which register with implementation of <code>global.KMeans</code>
 * to be notified of significant events during clustering.
 */
public interface KMeansListener {

    /**
     * A message has been received.
     * 
     * @param message
     */
    public void kmeansMessage(String message);
    
    /**
     * global.KMeans is complete.
     * 
     * @param clusters the output of clustering.
     * @param executionTime the time in milliseconds taken to cluster.
     */
    public void kmeansComplete(Cluster[] clusters, long executionTime);
    
    /**
     * An error occurred during global.KMeans clustering.
     * 
     * @param t
     */
    public void kmeansError(Throwable t);
    
}
