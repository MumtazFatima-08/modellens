import { useNavigate } from "react-router-dom";
import { FolderClock } from "lucide-react";
import { Card, SectionTitle, EmptyState } from "../components/ui";
import { getHistory } from "../lib/history";

export default function History() {
  const navigate = useNavigate();
  const history = getHistory();

  return (
    <div>
      <SectionTitle sublabel="Stored locally in your browser for this MVP — swap for a real database in a production deployment.">
        Investigation history
      </SectionTitle>
      {history.length === 0 ? (
        <EmptyState icon={<FolderClock size={22} />} title="No investigations yet" description="Investigations you run will appear here." />
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          {history.map((h) => (
            <Card key={h.id} className="cursor-pointer transition-colors hover:bg-white/5" >
              <div onClick={() => navigate(`/investigations/${h.id}`)}>
                <p className="font-display text-sm font-semibold capitalize text-white">{h.label}</p>
                <p className="font-mono mt-1 text-xs text-white/40">{h.id}</p>
                <p className="mt-1 text-xs text-white/40">{new Date(h.createdAt).toLocaleString()}</p>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
