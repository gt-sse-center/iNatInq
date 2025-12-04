import { useEffect, useState } from "react";

export default function TermsModal() {
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    // This only runs on the client side.
    // Check localStorage to see if the user has already accepted terms.
    const hasAcceptedTerms = localStorage.getItem("termsAccepted");
    if (!hasAcceptedTerms) {
      setShowModal(true);
    }
  }, []);

  const handleAgree = () => {
    // Set the flag in localStorage so we don't show the modal again.
    localStorage.setItem("termsAccepted", "true");
    setShowModal(false);
  };

  // If showModal is false, we don't render anything.
  if (!showModal) return null;

  return (
    <div className="fixed z-10 top-0 left-0 w-full h-full bg-gray-900 bg-opacity-50 flex items-center justify-center">
      <div className="bg-inatwhite px-8 py-6 rounded-lg shadow-lg max-w-[600px] text-center">
        <h2 className="font-bold">Terms of Use</h2>
        <p>By using this website, you agree to the following terms:</p>
        <div className="text-left bg-slate-100 p-4 rounded-lg text-xs text-slate-700 mt-4 overflow-scroll max-h-[300px] mb-4">
          <h2 className="font-bold text-sm mb-2">Informed Consent</h2>
          <p>
            This search tool is part of a MIT scientific research project that
            seeks to study image understanding and train and analyze automated
            image processing algorithms. As part of this study, you may complete
            a number of short queries as well as annotation tasks such as image
            tagging. Your decision to complete this task is voluntary. We do not
            anticipate any participation-related risk to you greater than those
            ordinarily encountered in daily life. Participation in the task will
            not be compensated. There are no anticipated direct benefits for you
            as a participant, though you will have the option to export your
            queries and labels for further personal or research use. There is no
            way for us to identify you. The only information we will have, in
            addition to your responses, is the time at which you completed the
            task, and a session identifier. The results of our analysis may be
            presented at scientific meetings or published in scientific
            journals, and the anonymous data may be shared for future research
            projects. If you have any pertinent questions about the research you
            may contact Sara Beery (beery@mit.edu). If you feel you have been
            treated unfairly, or you have questions regarding your rights as a
            research subject, you may contact the Chairman of the Committee on
            the Use of Humans as Experimental Subjects, M.I.T., Room E25-143b,
            77 Massachusetts Ave, Cambridge, MA 02139, phone 1-617-253-6787.
            Clicking on the 'SUBMIT' button on the bottom of this page indicates
            that you are at least 18 years of age and agree to complete this
            task voluntarily.
          </p>

          <h2 className="font-bold text-sm mt-6 mb-2">
            1. Acceptance of Terms
          </h2>
          <p>
            By accessing or using this tool (the "Service"), you ("User") agree
            to be bound by these Terms of Use ("Terms"). If you do not agree to
            these Terms, you must not use the Service.
          </p>

          <h2 className="font-bold text-sm mt-6 mb-2">
            2. Collection and Use of Data
          </h2>
          <h3 className="font-semibold mt-3 mb-1">Submission Data</h3>
          <ul className="list-disc ml-4">
            <li>
              The Service may collect and store User-submitted queries,
              timestamps, and anonymized session identifiers ("Submission
              Data").
            </li>
            <li>
              The Submission Data shall not be used to personally identify any
              individual.
            </li>
          </ul>
          <h3 className="font-semibold mt-3 mb-1">
            Positive and Negative Labels
          </h3>
          <ul className="list-disc ml-4">
            <li>
              User consents to the collection and retention of any positive or
              negative labels applied to their submissions for future
              methodological research, including improving the Service.
            </li>
            <li>
              All such data will be anonymized and used in accordance with these
              Terms.
            </li>
          </ul>

          <h2 className="font-bold text-sm mt-6 mb-2">
            3. Image Data and Licensing
          </h2>
          <h3 className="font-semibold mt-3 mb-1">
            No Ownership of Image Data
          </h3>
          <ul className="list-disc ml-4">
            <li>
              The Service does not claim any ownership rights in the images
              displayed or otherwise presented to the User through the Service
              ("Images").
            </li>
            <li>
              The Service does not host or store any Image files on its servers,
              nor does it retain the metadata or license information associated
              with such Images.
            </li>
          </ul>
          <h2 className="font-bold text-sm mt-6 mb-2">
            4. Responsibility for Content
          </h2>
          <ul className="list-disc ml-4">
            <li>
              Users acknowledge and agree that the Service is not responsible or
              liable for the content of any Images shown through the Service.
            </li>
            <li>
              The Service does not endorse or guarantee the accuracy, legality,
              or appropriateness of any Image or content.
            </li>
          </ul>

          <h2 className="font-bold text-sm mt-6 mb-2">
            5. Intended Use and Ethics
          </h2>
          <ul className="list-disc ml-4">
            <li>
              The Service is intended solely for scientific and research-related
              purposes.
            </li>
            <li>
              The Service disclaims all responsibility and liability for any
              misuse or unintended use of the Service.
            </li>
          </ul>

          <h2 className="font-bold text-sm mt-6 mb-2">
            6. Disclaimer of Warranties
          </h2>
          <ul className="list-disc ml-4">
            <li>
              The Service is provided on an "as is" and "as available" basis
              without warranties of any kind, either express or implied.
            </li>
            <li>
              To the fullest extent permitted by law, the Service disclaims all
              warranties, including, without limitation, warranties of
              merchantability, fitness for a particular purpose, and
              non-infringement.
            </li>
            <li>
              The Service does not guarantee the accuracy, completeness, or
              usefulness of any information or content provided.
            </li>
          </ul>

          <h2 className="font-bold text-sm mt-6 mb-2">
            7. Limitation of Liability
          </h2>
          <ul className="list-disc ml-4">
            <li>
              To the maximum extent permitted by applicable law, in no event
              shall the Service, its directors, employees, or agents be liable
              for any direct, indirect, incidental, special, consequential, or
              punitive damages arising out of or related to the use of the
              Service.
            </li>
            <li>
              This includes, but is not limited to, damages for loss of profits,
              goodwill, use, data, or other intangible losses.
            </li>
          </ul>

          <h2 className="font-bold text-sm mt-6 mb-2">8. Contact Us</h2>
          <ul className="list-disc ml-4">
            <li>
              If you have any questions about these Terms, please contact us at
              <a
                href="mailto:evendrow@mit.edu"
                className="text-blue-500 hover:underline mx-1"
              >
                evendrow@mit.edu
              </a>
            </li>
          </ul>
        </div>
        <button
          className="text-md px-3 py-1 rounded border border-slate-600 hover:bg-slate-300 text-slate-700 font-medium"
          onClick={() => {
            handleAgree();
          }}
          type="button"
        >
          I agree
        </button>
      </div>
    </div>
  );
}
