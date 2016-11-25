package global;

/**
 * Simple K-Means clustering interface.
 */
public interface KMeans extends Runnable {
    
    /** 
     * Adds a global.KMeansListener to be notified of significant happenings.
     * 
     * @param l  the listener to be added.
     */
    public void addKMeansListener(KMeansListener l);
    
    /**
     * Removes a global.KMeansListener from the listener list.
     * 
     * @param l the listener to be removed.
     */
    public void removeKMeansListener(KMeansListener l);
    
    /**
     * Get the clusters computed by the algorithm.  This method should
     * not be called until clustering has completed successfully.
     * 
     * @return an array of global.Cluster objects.
     */
    public Cluster[] getClusters();
 
}
